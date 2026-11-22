// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use crate::errors::ParquetError;
use crate::errors::Result;

/// A buffer that supports padding with nulls
pub trait ValuesBuffer {
    /// Create a new buffer with capacity for at least `capacity` logical values
    fn with_capacity(capacity: usize) -> Self;

    /// Reserve capacity for `additional` more logical values.
    ///
    /// Callers should use this once the level decoders have determined how
    /// many output slots are actually needed for the next read.
    fn reserve_exact(&mut self, additional: usize);

    /// If a column contains nulls, more level data may be read than value data, as null
    /// values are not encoded. Therefore, first the levels data is read, the null count
    /// determined, and then the corresponding number of values read to a [`ValuesBuffer`].
    ///
    /// It is then necessary to move this values data into positions that correspond to
    /// the non-null level positions. This is what this method does.
    ///
    /// It is provided with:
    ///
    /// - `read_offset` - the offset in [`ValuesBuffer`] to start null padding from
    /// - `values_read` - the number of values read
    /// - `levels_read` - the number of levels read
    /// - `valid_mask` - a packed mask of valid levels
    ///
    /// Returns an error if the inputs are inconsistent, for example because the
    /// decoded data was corrupt. This must not panic on such input.
    fn pad_nulls(
        &mut self,
        read_offset: usize,
        values_read: usize,
        levels_read: usize,
        valid_mask: &[u8],
    ) -> Result<()>;
}

impl<T: Copy + Default> ValuesBuffer for Vec<T> {
    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }

    fn reserve_exact(&mut self, additional: usize) {
        Vec::reserve_exact(self, additional)
    }

    fn pad_nulls(
        &mut self,
        read_offset: usize,
        values_read: usize,
        levels_read: usize,
        valid_mask: &[u8],
    ) -> Result<()> {
        // Walks the mask's 64-bit words from the top and moves each run of set
        // bits with one copy, instead of one move per set bit. Same result as
        // pairing `iter_set_bits_rev` with the values in reverse: stops when
        // the values run out or a value is already at its level position.
        let end = read_offset + levels_read;
        self.resize(end, T::default());
        let mut remaining = values_read;
        let mut src_end = read_offset + values_read;
        let mut w = valid_mask.len().div_ceil(8);
        while w > 0 && remaining > 0 {
            w -= 1;
            let start = w * 8;
            let bytes = &valid_mask[start..valid_mask.len().min(start + 8)];
            let mut word = [0u8; 8];
            word[..bytes.len()].copy_from_slice(bytes);
            let mut c = u64::from_le_bytes(word);
            while c != 0 && remaining > 0 {
                let hi = 63 - c.leading_zeros() as usize;
                let ones = (!(c << (63 - hi))).leading_zeros() as usize;
                let lo = hi + 1 - ones;
                let run_end = w * 64 + hi + 1;
                if run_end > end {
                    return Err(general_err!(
                        "valid mask bit {} is beyond {} levels",
                        run_end - 1,
                        end
                    ));
                }
                let n = ones.min(remaining);
                let dst = run_end - n;
                let src = src_end - n;
                if dst <= src {
                    return Ok(());
                }
                if n == 1 {
                    self[dst] = self[src];
                } else {
                    self.copy_within(src..src_end, dst);
                }
                src_end = src;
                remaining -= n;
                c &= (1u64 << lo).wrapping_sub(1);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::buffer::bit_util::iter_set_bits_rev;

    /// The per-value version this replaces: pair the values in reverse with
    /// the set bits in reverse, stop when a value is already in place.
    fn reference(
        v: &mut Vec<u64>,
        read_offset: usize,
        values_read: usize,
        levels_read: usize,
        valid_mask: &[u8],
    ) {
        v.resize(read_offset + levels_read, 0);
        let values_range = read_offset..read_offset + values_read;
        for (value_pos, level_pos) in values_range.rev().zip(iter_set_bits_rev(valid_mask)) {
            if level_pos <= value_pos {
                break;
            }
            v[level_pos] = v[value_pos];
        }
    }

    struct XorShift(u64);

    impl XorShift {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n.max(1)
        }
    }

    #[test]
    fn pad_nulls_matches_the_per_value_version() {
        let mut r = XorShift(0x2545_f491_4f6c_dd1d);
        for case in 0..20_000 {
            let read_offset = r.below(150) as usize;
            let levels_read = r.below(300) as usize;
            let end = read_offset + levels_read;
            // Null rates from none to all, scattered or in runs.
            let null_per_mille = [0, 10, 40, 100, 300, 500, 900, 1000][r.below(8) as usize];
            let runs = r.below(2) == 0;
            let mut mask = vec![0u8; end.div_ceil(8) + r.below(3) as usize];
            for i in 0..read_offset {
                if r.below(2) == 0 {
                    mask[i / 8] |= 1 << (i % 8);
                }
            }
            let mut valid = 0;
            let mut state = true;
            for i in read_offset..end {
                let set = if runs {
                    if r.below(1000) < 50 {
                        state = !state;
                    }
                    state && r.below(1000) >= null_per_mille / 4
                } else {
                    r.below(1000) >= null_per_mille
                };
                if set {
                    mask[i / 8] |= 1 << (i % 8);
                    valid += 1;
                }
            }
            // Mostly consistent counts, sometimes fewer values than set bits.
            let values_read = if r.below(10) == 0 {
                r.below(levels_read as u64 + 1) as usize
            } else {
                valid
            };
            let mut start: Vec<u64> = (0..read_offset as u64).map(|x| x + 1_000_000).collect();
            start.extend((0..values_read as u64).map(|x| x + 1));
            let mut expected = start.clone();
            reference(&mut expected, read_offset, values_read, levels_read, &mask);
            let mut actual = start;
            actual
                .pad_nulls(read_offset, values_read, levels_read, &mask)
                .unwrap();
            assert_eq!(actual, expected, "case {case}");
        }
    }

    #[test]
    fn pad_nulls_refuses_a_valid_bit_past_the_levels() {
        let mut v: Vec<u64> = vec![1, 2];
        assert!(v.pad_nulls(0, 2, 2, &[0b101]).is_err());
    }
}

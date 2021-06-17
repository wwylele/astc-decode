// Copyright 2021 Weiyi Wang
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The code is ported from an ASTC decoder in C++ with a few bug fixes.
// Here is the copyright notice from the original code:
//
// Copyright 2016 The University of North Carolina at Chapel Hill
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Please send all BUG REPORTS to <pavel@cs.unc.edu>.
// <http://gamma.cs.unc.edu/FasTC/>

#![allow(clippy::many_single_char_names)]

use std::convert::TryFrom;
use std::io::*;

struct InputBitStream {
    data: u128,
    bits_read: u32,
}

impl InputBitStream {
    fn new(data: u128) -> InputBitStream {
        InputBitStream { data, bits_read: 0 }
    }

    fn get_bits_read(&self) -> u32 {
        self.bits_read
    }

    fn read_bit(&mut self) -> u32 {
        self.read_bits(1)
    }

    fn read_bits(&mut self, n_bits: u32) -> u32 {
        assert!(n_bits <= 32);
        self.read_bits128(n_bits) as u32
    }

    fn read_bits128(&mut self, n_bits: u32) -> u128 {
        self.bits_read += n_bits;
        assert!(self.bits_read <= 128);
        let ret = self.data & ((1 << n_bits) - 1);
        self.data >>= n_bits;
        ret
    }
}

struct Bits(u32);

impl Bits {
    fn get(&self, pos: u32) -> u32 {
        (self.0 >> pos) & 1
    }

    fn range(&self, start: u32, end: u32) -> u32 {
        let mask = (1 << (end - start + 1)) - 1;
        (self.0 >> start) & mask
    }
}

#[derive(PartialEq, Eq)]
enum IntegerEncoding {
    JustBits,
    Quint(u32),
    Trit(u32),
}

struct IntegerEncodedValue {
    encoding: IntegerEncoding,
    num_bits: u32,
    bit_value: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IntegerEncodingType {
    JustBits,
    Quint,
    Trit,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct IntegerEncodingFormat {
    encoding: IntegerEncodingType,
    num_bits: u32,
}

impl IntegerEncodingFormat {
    // Returns the number of bits required to encode nVals values.
    fn get_bit_length(&self, n_vals: u32) -> u32 {
        let mut total_bits = self.num_bits * n_vals;
        match self.encoding {
            IntegerEncodingType::JustBits => (),
            IntegerEncodingType::Trit => total_bits += (n_vals * 8 + 4) / 5,
            IntegerEncodingType::Quint => total_bits += (n_vals * 7 + 2) / 3,
        }
        total_bits
    }
}

fn decode_trit_block(
    bits: &mut InputBitStream,
    result: &mut Vec<IntegerEncodedValue>,
    bits_per_value: u32,
) {
    // Implement the algorithm in section c.2.12
    let mut m = [0u32; 5];
    let mut t = [0u32; 5];
    let mut tt: u32;

    // Read the trit encoded block according to
    // table c.2.14
    m[0] = bits.read_bits(bits_per_value);
    tt = bits.read_bits(2);
    m[1] = bits.read_bits(bits_per_value);
    tt |= bits.read_bits(2) << 2;
    m[2] = bits.read_bits(bits_per_value);
    tt |= (bits.read_bit()) << 4;
    m[3] = bits.read_bits(bits_per_value);
    tt |= bits.read_bits(2) << 5;
    m[4] = bits.read_bits(bits_per_value);
    tt |= (bits.read_bit()) << 7;

    let c: u32;

    let tb = Bits(tt);
    if tb.range(2, 4) == 7 {
        c = (tb.range(5, 7) << 2) | tb.range(0, 1);
        t[3] = 2;
        t[4] = 2;
    } else {
        c = tb.range(0, 4);
        if tb.range(5, 6) == 3 {
            t[4] = 2;
            t[3] = tb.get(7);
        } else {
            t[4] = tb.get(7);
            t[3] = tb.range(5, 6);
        }
    }

    let cb = Bits(c);
    if cb.range(0, 1) == 3 {
        t[2] = 2;
        t[1] = cb.get(4);
        t[0] = (cb.get(3) << 1) | (cb.get(2) & !cb.get(3));
    } else if cb.range(2, 3) == 3 {
        t[2] = 2;
        t[1] = 2;
        t[0] = cb.range(0, 1);
    } else {
        t[2] = cb.get(4);
        t[1] = cb.range(2, 3);
        t[0] = (cb.get(1) << 1) | (cb.get(0) & !cb.get(1));
    }

    for i in 0..5 {
        let bit_value = m[i];
        let trit_value = t[i];
        result.push(IntegerEncodedValue {
            encoding: IntegerEncoding::Trit(trit_value),
            bit_value,
            num_bits: bits_per_value,
        })
    }
}

fn decode_quint_block(
    bits: &mut InputBitStream,
    result: &mut Vec<IntegerEncodedValue>,
    bits_per_value: u32,
) {
    // Implement the algorithm in section c.2.12
    let mut m = [0u32; 3];
    let mut q = [0u32; 3];
    let mut qq: u32;

    // Read the trit encoded block according to
    // table c.2.15
    m[0] = bits.read_bits(bits_per_value);
    qq = bits.read_bits(3);
    m[1] = bits.read_bits(bits_per_value);
    qq |= bits.read_bits(2) << 3;
    m[2] = bits.read_bits(bits_per_value);
    qq |= bits.read_bits(2) << 5;

    let qb = Bits(qq);
    if qb.range(1, 2) == 3 && qb.range(5, 6) == 0 {
        q[0] = 4;
        q[1] = 4;
        q[2] = (qb.get(0) << 2) | ((qb.get(4) & !qb.get(0)) << 1) | (qb.get(3) & !qb.get(0));
    } else {
        let c;
        if qb.range(1, 2) == 3 {
            q[2] = 4;
            c = (qb.range(3, 4) << 3) | ((!qb.range(5, 6) & 3) << 1) | qb.get(0);
        } else {
            q[2] = qb.range(5, 6);
            c = qb.range(0, 4);
        }

        let cb = Bits(c);
        if cb.range(0, 2) == 5 {
            q[1] = 4;
            q[0] = cb.range(3, 4);
        } else {
            q[1] = cb.range(3, 4);
            q[0] = cb.range(0, 2);
        }
    }

    for i in 0..3 {
        let bit_value = m[i];
        let quint_value = q[i];
        result.push(IntegerEncodedValue {
            encoding: IntegerEncoding::Quint(quint_value),
            bit_value,
            num_bits: bits_per_value,
        })
    }
}

// Returns a new instance of this struct that corresponds to the
// can take no more than maxval values
const fn create_encoding(mut max_val: u32) -> IntegerEncodingFormat {
    while max_val > 0 {
        let check = max_val + 1;

        // Is max_val a power of two?
        if (check & (check - 1)) == 0 {
            return IntegerEncodingFormat {
                encoding: IntegerEncodingType::JustBits,
                num_bits: max_val.count_ones(),
            };
        }

        // Is max_val of the type 3*2^n - 1?
        if (check % 3 == 0) && ((check / 3) & ((check / 3) - 1)) == 0 {
            return IntegerEncodingFormat {
                encoding: IntegerEncodingType::Trit,
                num_bits: (check / 3 - 1).count_ones(),
            };
        }

        // Is max_val of the type 5*2^n - 1?
        if (check % 5 == 0) && ((check / 5) & ((check / 5) - 1)) == 0 {
            return IntegerEncodingFormat {
                encoding: IntegerEncodingType::Quint,
                num_bits: (check / 5 - 1).count_ones(),
            };
        }

        // Apparently it can't be represented with a bounded integer sequence...
        // just iterate.
        max_val -= 1;
    }
    IntegerEncodingFormat {
        encoding: IntegerEncodingType::JustBits,
        num_bits: 0,
    }
}

static ENCODINGS_VALUES: [IntegerEncodingFormat; 256] = {
    let mut result = [IntegerEncodingFormat {
        encoding: IntegerEncodingType::JustBits,
        num_bits: 0,
    }; 256];
    let mut i = 0;
    while i < 256 {
        result[i as usize] = create_encoding(i);
        i += 1;
    }
    result
};

// Fills result with the values that are encoded in the given
// bitstream. We must know beforehand what the maximum possible
// value is, and how many values we're decoding.
fn decode_integer_sequence(
    bits: &mut InputBitStream,
    max_range: u32,
    n_values: u32,
) -> Vec<IntegerEncodedValue> {
    let mut result = vec![];

    // Determine encoding parameters
    let val = ENCODINGS_VALUES[max_range as usize];

    // Start decoding
    let mut n_vals_decoded = 0;
    while n_vals_decoded < n_values {
        match val.encoding {
            IntegerEncodingType::Quint => {
                decode_quint_block(bits, &mut result, val.num_bits);
                n_vals_decoded += 3;
            }

            IntegerEncodingType::Trit => {
                decode_trit_block(bits, &mut result, val.num_bits);
                n_vals_decoded += 5;
            }
            IntegerEncodingType::JustBits => {
                let bit_value = bits.read_bits(val.num_bits);
                result.push(IntegerEncodedValue {
                    num_bits: val.num_bits,
                    bit_value,
                    encoding: IntegerEncoding::JustBits,
                });
                n_vals_decoded += 1;
            }
        }
    }

    result
}

struct TexelWeightParams {
    width: u32,
    height: u32,
    is_dual_plane: bool,
    max_weight: u32,
    is_error: bool,
    void_extent_ldr: bool,
    void_extent_hdr: bool,
}

impl Default for TexelWeightParams {
    fn default() -> Self {
        TexelWeightParams {
            width: 0,
            height: 0,
            is_dual_plane: false,
            max_weight: 0,
            is_error: false,
            void_extent_ldr: false,
            void_extent_hdr: false,
        }
    }
}

impl TexelWeightParams {
    fn get_packed_bit_size(&self) -> u32 {
        ENCODINGS_VALUES[self.max_weight as usize].get_bit_length(self.get_num_weight_values())
    }

    fn get_num_weight_values(&self) -> u32 {
        let mut ret = self.width * self.height;
        if self.is_dual_plane {
            ret *= 2;
        }
        ret
    }
}

fn decode_block_info(strm: &mut InputBitStream) -> TexelWeightParams {
    let mut params = TexelWeightParams::default();

    // Read the entire block mode all at once
    let mode_bits = strm.read_bits(11);

    // Does this match the void extent block mode?
    if (mode_bits & 0x01FF) == 0x1FC {
        if mode_bits & 0x200 != 0 {
            params.void_extent_hdr = true;
        } else {
            params.void_extent_ldr = true;
        }

        // Next two bits must be one.
        if (mode_bits & 0x400) == 0 || strm.read_bit() == 0 {
            params.is_error = true;
        }

        return params;
    }

    // First check if the last four bits are zero
    if (mode_bits & 0xF) == 0 {
        params.is_error = true;
        return params;
    }

    // If the last two bits are zero, then if bits
    // [6-8] are all ones, this is also reserved.
    if (mode_bits & 0x3) == 0 && (mode_bits & 0x1C0) == 0x1C0 {
        params.is_error = true;
        return params;
    }

    // Otherwise, there is no error... Figure out the layout
    // of the block mode. Layout is determined by a number
    // between 0 and 9 corresponding to table c.2.8 of the
    // ASTC spec.
    let layout;

    if (mode_bits & 0x1) != 0 || (mode_bits & 0x2) != 0 {
        // layout is in [0-4]
        if (mode_bits & 0x8) != 0 {
            // layout is in [2-4]
            if (mode_bits & 0x4) != 0 {
                // layout is in [3-4]
                if (mode_bits & 0x100) != 0 {
                    layout = 4;
                } else {
                    layout = 3;
                }
            } else {
                layout = 2;
            }
        } else {
            // layout is in [0-1]
            if (mode_bits & 0x4) != 0 {
                layout = 1;
            } else {
                layout = 0;
            }
        }
    } else {
        // layout is in [5-9]
        if (mode_bits & 0x100) != 0 {
            // layout is in [7-9]
            if (mode_bits & 0x80) != 0 {
                // layout is in [7-8]
                assert!((mode_bits & 0x40) == 0);
                if (mode_bits & 0x20) != 0 {
                    layout = 8;
                } else {
                    layout = 7;
                }
            } else {
                layout = 9;
            }
        } else {
            // layout is in [5-6]
            if (mode_bits & 0x80) != 0 {
                layout = 6;
            } else {
                layout = 5;
            }
        }
    }

    assert!(layout < 10);

    // Determine R
    let mut r = (mode_bits & 0x10) >> 4;
    if layout < 5 {
        r |= (mode_bits & 0x3) << 1;
    } else {
        r |= (mode_bits & 0xC) >> 1;
    }
    assert!((2..=7).contains(&r));

    // Determine width & height
    match layout {
        0 => {
            let a = (mode_bits >> 5) & 0x3;
            let b = (mode_bits >> 7) & 0x3;
            params.width = b + 4;
            params.height = a + 2;
        }

        1 => {
            let a = (mode_bits >> 5) & 0x3;
            let b = (mode_bits >> 7) & 0x3;
            params.width = b + 8;
            params.height = a + 2;
        }

        2 => {
            let a = (mode_bits >> 5) & 0x3;
            let b = (mode_bits >> 7) & 0x3;
            params.width = a + 2;
            params.height = b + 8;
        }

        3 => {
            let a = (mode_bits >> 5) & 0x3;
            let b = (mode_bits >> 7) & 0x1;
            params.width = a + 2;
            params.height = b + 6;
        }

        4 => {
            let a = (mode_bits >> 5) & 0x3;
            let b = (mode_bits >> 7) & 0x1;
            params.width = b + 2;
            params.height = a + 2;
        }

        5 => {
            let a = (mode_bits >> 5) & 0x3;
            params.width = 12;
            params.height = a + 2;
        }

        6 => {
            let a = (mode_bits >> 5) & 0x3;
            params.width = a + 2;
            params.height = 12;
        }

        7 => {
            params.width = 6;
            params.height = 10;
        }

        8 => {
            params.width = 10;
            params.height = 6;
        }

        9 => {
            let a = (mode_bits >> 5) & 0x3;
            let b = (mode_bits >> 9) & 0x3;
            params.width = a + 6;
            params.height = b + 6;
        }

        _ => panic!("Don't know this layout..."),
    }

    // Determine whether or not we're using dual planes
    // and/or high precision layouts.
    let dp = (layout != 9) && (mode_bits & 0x400) != 0;
    let p = (layout != 9) && (mode_bits & 0x200) != 0;

    if p {
        const MAX_WEIGHTS: [u32; 6] = [9, 11, 15, 19, 23, 31];
        params.max_weight = MAX_WEIGHTS[(r - 2) as usize];
    } else {
        const MAX_WEIGHTS: [u32; 6] = [1, 2, 3, 4, 5, 7];
        params.max_weight = MAX_WEIGHTS[(r - 2) as usize];
    }

    params.is_dual_plane = dp;

    params
}

fn fill_void_extent_ldr<F: FnMut(u32, u32, [u8; 4])>(
    strm: &mut InputBitStream,
    writer: &mut F,
    block_width: u32,
    block_height: u32,
) {
    // Don't actually care about the void extent, just read the bits...
    for _ in 0..4 {
        strm.read_bits(13);
    }

    // Decode the RGBA components and renormalize them to the range [0, 255]
    let r = strm.read_bits(16) >> 8;
    let g = strm.read_bits(16) >> 8;
    let b = strm.read_bits(16) >> 8;
    let a = strm.read_bits(16) >> 8;

    for j in 0..block_height {
        for i in 0..block_width {
            writer(i, j, [r as u8, g as u8, b as u8, a as u8]);
        }
    }
}

fn fill_error<F: FnMut(u32, u32, [u8; 4])>(writer: &mut F, block_width: u32, block_height: u32) {
    for j in 0..block_height {
        for i in 0..block_width {
            writer(i, j, [0xFF, 0, 0xFF, 0xFF]);
        }
    }
}

// Replicates low num_bits such that [(to_bit - 1):(to_bit - num_bits)]
// is the same as [(num_bits - 1):0] and repeats all the way down.
fn replicate(val: u32, mut num_bits: u32, to_bit: u32) -> u32 {
    if num_bits == 0 {
        return 0;
    }
    if to_bit == 0 {
        return 0;
    }
    let v = val & ((1 << num_bits) - 1);
    let mut res = v;
    let mut reslen = num_bits;
    while reslen < to_bit {
        let mut comp = 0;
        if num_bits > to_bit - reslen {
            let newshift = to_bit - reslen;
            comp = num_bits - newshift;
            num_bits = newshift;
        }
        res <<= num_bits;
        res |= v >> comp;
        reslen += num_bits;
    }
    res
}

fn decode_color_values(
    data: u128,
    modes: &[u32],
    n_partitions: usize,
    n_bits_for_color_data: u32,
) -> [u8; 32] {
    let mut out = [0; 32]; // Four values, two endpoints, four maximum paritions

    // First figure out how many color values we have
    let n_values = modes[0..n_partitions]
        .iter()
        .map(|m| ((m >> 2) + 1) << 1)
        .sum();

    // Then based on the number of values and the remaining number of bits,
    // figure out the max value for each of them...
    let mut range = 256;
    range -= 1;
    while range > 0 {
        let val = ENCODINGS_VALUES[range];
        let bit_length = val.get_bit_length(n_values);
        if bit_length <= n_bits_for_color_data {
            // Find the smallest possible range that matches the given encoding
            range -= 1;
            while range > 0 {
                let newval = ENCODINGS_VALUES[range];
                if newval != val {
                    break;
                }
                range -= 1;
            }

            // Return to last matching range.
            range += 1;
            break;
        }
        range -= 1;
    }

    // We now have enough to decode our integer sequence.
    let mut color_stream = InputBitStream::new(data);
    let decoded_color_values = decode_integer_sequence(&mut color_stream, range as u32, n_values);

    // Once we have the decoded values, we need to dequantize them to the 0-255 range
    // This procedure is outlined in ASTC spec c.2.13
    let mut out_idx = 0;
    for val in decoded_color_values {
        // Have we already decoded all that we need?
        if out_idx >= n_values {
            break;
        }

        let bitlen = val.num_bits;
        let bitval = val.bit_value;

        assert!(bitlen >= 1);

        // a is just the lsb replicated 9 times.
        let a = (bitval & 1) * 0x1FF;
        let mut b = 0;
        let mut c = 0;
        let mut d = 0;

        match val.encoding {
            // replicate bits
            IntegerEncoding::JustBits => {
                out[out_idx as usize] = u8::try_from(replicate(bitval, bitlen, 8)).unwrap();
                out_idx += 1;
            }

            // Use algorithm in c.2.13
            IntegerEncoding::Trit(trit_value) => {
                d = trit_value;

                match bitlen {
                    1 => {
                        c = 204;
                    }

                    2 => {
                        c = 93;
                        // b = b000b0bb0
                        let x = (bitval >> 1) & 1;
                        b = (x << 8) | (x << 4) | (x << 2) | (x << 1);
                    }

                    3 => {
                        c = 44;
                        // b = cb000cbcb
                        let cb = (bitval >> 1) & 3;
                        b = (cb << 7) | (cb << 2) | cb;
                    }

                    4 => {
                        c = 22;
                        // b = dcb000dcb
                        let dcb = (bitval >> 1) & 7;
                        b = (dcb << 6) | dcb;
                    }

                    5 => {
                        c = 11;
                        // b = edcb000ed
                        let edcb = (bitval >> 1) & 0xF;
                        b = (edcb << 5) | (edcb >> 2);
                    }

                    6 => {
                        c = 5;
                        // b = fedcb000f
                        let fedcb = (bitval >> 1) & 0x1F;
                        b = (fedcb << 4) | (fedcb >> 4);
                    }

                    _ => panic!("Unsupported trit encoding for color values!"),
                }
            }

            IntegerEncoding::Quint(quint_value) => {
                d = quint_value;

                match bitlen {
                    1 => {
                        c = 113;
                    }

                    2 => {
                        c = 54;
                        // b = b0000bb00
                        let x = (bitval >> 1) & 1;
                        b = (x << 8) | (x << 3) | (x << 2);
                    }

                    3 => {
                        c = 26;
                        // b = cb0000cbc
                        let cb = (bitval >> 1) & 3;
                        b = (cb << 7) | (cb << 1) | (cb >> 1);
                    }

                    4 => {
                        c = 13;
                        // b = dcb0000dc
                        let dcb = (bitval >> 1) & 7;
                        b = (dcb << 6) | (dcb >> 1);
                    }

                    5 => {
                        c = 6;
                        // b = edcb0000e
                        let edcb = (bitval >> 1) & 0xF;
                        b = (edcb << 5) | (edcb >> 3);
                    }

                    _ => panic!("Unsupported quint encoding for color values!"),
                }
            }
        } // switch(val.encoding)

        if val.encoding != IntegerEncoding::JustBits {
            let mut t = d * c + b;
            t ^= a;
            t = (a & 0x80) | (t >> 2);
            out[out_idx as usize] = u8::try_from(t).unwrap();
            out_idx += 1;
        }
    }
    out
}

fn unquantize_texel_weight(val: &IntegerEncodedValue) -> u32 {
    let bitval = val.bit_value;
    let bitlen = val.num_bits;

    let a = (bitval & 1) * 0x7F;
    let mut b = 0;
    let mut c = 0;
    let mut d = 0;

    let mut result = 0;
    match val.encoding {
        IntegerEncoding::JustBits => {
            result = replicate(bitval, bitlen, 6);
        }

        IntegerEncoding::Trit(trit_value) => {
            d = trit_value;
            assert!(d < 3);

            match bitlen {
                0 => {
                    result = [0, 32, 63][d as usize];
                }

                1 => {
                    c = 50;
                }

                2 => {
                    c = 23;
                    let x = (bitval >> 1) & 1;
                    b = (x << 6) | (x << 2) | x;
                }

                3 => {
                    c = 11;
                    let cb = (bitval >> 1) & 3;
                    b = (cb << 5) | cb;
                }

                _ => panic!("Invalid trit encoding for texel weight"),
            }
        }

        IntegerEncoding::Quint(quint_value) => {
            d = quint_value;
            assert!(d < 5);

            match bitlen {
                0 => {
                    result = [0, 16, 32, 47, 63][d as usize];
                }

                1 => {
                    c = 28;
                }

                2 => {
                    c = 13;
                    let x = (bitval >> 1) & 1;
                    b = (x << 6) | (x << 1);
                }

                _ => panic!("Invalid quint encoding for texel weight"),
            }
        }
    }

    if val.encoding != IntegerEncoding::JustBits && bitlen > 0 {
        // Decode the value...
        result = d * c + b;
        result ^= a;
        result = (a & 0x20) | (result >> 2);
    }

    assert!(result < 64);

    // Change from [0,63] to [0,64]
    if result > 32 {
        result += 1;
    }

    result
}

fn unquantize_texel_weights(
    weights: &[IntegerEncodedValue],
    params: &TexelWeightParams,
    block_width: u32,
    block_height: u32,
) -> [[u32; 144]; 2] {
    // Blocks can be at most 12x12, so we can have as many as 144 weights
    let mut out = [[0; 144]; 2];
    let mut unquantized = [[0; 144]; 2];

    let plane_scale = if params.is_dual_plane { 2 } else { 1 };

    for (i, weight_chunk) in weights.chunks_exact(plane_scale).enumerate() {
        for (plane, weight) in weight_chunk.iter().enumerate() {
            unquantized[plane][i] = unquantize_texel_weight(weight);
        }
    }

    // Do infill if necessary (Section c.2.18) ...
    let ds = (1024 + (block_width / 2)) / (block_width - 1);
    let dt = (1024 + (block_height / 2)) / (block_height - 1);

    for plane in 0..plane_scale {
        for t in 0..block_height {
            for s in 0..block_width {
                let cs = ds * s;
                let ct = dt * t;

                let gs = (cs * (params.width - 1) + 32) >> 6;
                let gt = (ct * (params.height - 1) + 32) >> 6;

                let js = gs >> 4;
                let fs = gs & 0xF;

                let jt = gt >> 4;
                let ft = gt & 0x0F;

                let w11 = (fs * ft + 8) >> 4;
                let w10 = ft - w11;
                let w01 = fs - w11;
                let w00 = 16 + w11 - fs - ft;

                let v0 = js + jt * params.width;

                let mut p00 = 0;
                let mut p01 = 0;
                let mut p10 = 0;
                let mut p11 = 0;

                if v0 < (params.width * params.height) {
                    p00 = unquantized[plane][v0 as usize];
                }

                if v0 + 1 < (params.width * params.height) {
                    p01 = unquantized[plane][(v0 + 1) as usize];
                }

                if v0 + params.width < (params.width * params.height) {
                    p10 = unquantized[plane][(v0 + params.width) as usize];
                }

                if v0 + params.width + 1 < (params.width * params.height) {
                    p11 = unquantized[plane][(v0 + params.width + 1) as usize];
                }

                out[plane][(t * block_width + s) as usize] =
                    (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11 + 8) >> 4;
            }
        }
    }
    out
}

// Transfers a bit as described in c.2.14
fn bit_transfer_signed(a: &mut i32, b: &mut i32) {
    *b >>= 1;
    *b |= *a & 0x80;
    *a >>= 1;
    *a &= 0x3F;
    if (*a & 0x20) != 0 {
        *a -= 0x40;
    }
}

// Partition selection functions as specified in
// c.2.21
fn hash52(p: u32) -> u32 {
    let mut p = std::num::Wrapping(p);
    p ^= p >> 15;
    p -= p << 17;
    p += p << 7;
    p += p << 4;
    p ^= p >> 5;
    p += p << 16;
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    p.0
}

fn select_partition(
    mut seed: u32,
    mut x: u32,
    mut y: u32,
    mut z: u32,
    partition_count: usize,
    small_block: bool,
) -> usize {
    if 1 == partition_count {
        return 0;
    }

    if small_block {
        x <<= 1;
        y <<= 1;
        z <<= 1;
    }

    seed += (partition_count as u32 - 1) * 1024;

    let rnum = hash52(seed);
    let mut seed1 = (rnum & 0xF) as u8;
    let mut seed2 = ((rnum >> 4) & 0xF) as u8;
    let mut seed3 = ((rnum >> 8) & 0xF) as u8;
    let mut seed4 = ((rnum >> 12) & 0xF) as u8;
    let mut seed5 = ((rnum >> 16) & 0xF) as u8;
    let mut seed6 = ((rnum >> 20) & 0xF) as u8;
    let mut seed7 = ((rnum >> 24) & 0xF) as u8;
    let mut seed8 = ((rnum >> 28) & 0xF) as u8;
    let mut seed9 = ((rnum >> 18) & 0xF) as u8;
    let mut seed10 = ((rnum >> 22) & 0xF) as u8;
    let mut seed11 = ((rnum >> 26) & 0xF) as u8;
    let mut seed12 = (((rnum >> 30) | (rnum << 2)) & 0xF) as u8;

    seed1 = seed1 * seed1;
    seed2 = seed2 * seed2;
    seed3 = seed3 * seed3;
    seed4 = seed4 * seed4;
    seed5 = seed5 * seed5;
    seed6 = seed6 * seed6;
    seed7 = seed7 * seed7;
    seed8 = seed8 * seed8;
    seed9 = seed9 * seed9;
    seed10 = seed10 * seed10;
    seed11 = seed11 * seed11;
    seed12 = seed12 * seed12;

    let sh1: i32;
    let sh2: i32;
    let sh3: i32;
    if seed & 1 != 0 {
        sh1 = if seed & 2 != 0 { 4 } else { 5 };
        sh2 = if partition_count == 3 { 6 } else { 5 };
    } else {
        sh1 = if partition_count == 3 { 6 } else { 5 };
        sh2 = if seed & 2 != 0 { 4 } else { 5 };
    }
    sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };

    seed1 >>= sh1;
    seed2 >>= sh2;
    seed3 >>= sh1;
    seed4 >>= sh2;
    seed5 >>= sh1;
    seed6 >>= sh2;
    seed7 >>= sh1;
    seed8 >>= sh2;
    seed9 >>= sh3;
    seed10 >>= sh3;
    seed11 >>= sh3;
    seed12 >>= sh3;

    let mut a = seed1 as u32 * x + seed2 as u32 * y + seed11 as u32 * z + (rnum >> 14);
    let mut b = seed3 as u32 * x + seed4 as u32 * y + seed12 as u32 * z + (rnum >> 10);
    let mut c = seed5 as u32 * x + seed6 as u32 * y + seed9 as u32 * z + (rnum >> 6);
    let mut d = seed7 as u32 * x + seed8 as u32 * y + seed10 as u32 * z + (rnum >> 2);

    a &= 0x3F;
    b &= 0x3F;
    c &= 0x3F;
    d &= 0x3F;

    if partition_count < 4 {
        d = 0;
    }

    if partition_count < 3 {
        c = 0;
    }

    if a >= b && a >= c && a >= d {
        0
    } else if b >= c && b >= d {
        1
    } else if c >= d {
        2
    } else {
        3
    }
}

fn select_2d_partition(
    seed: u32,
    x: u32,
    y: u32,
    partition_count: usize,
    small_block: bool,
) -> usize {
    select_partition(seed, x, y, 0, partition_count, small_block)
}

fn clamp_color(r: i32, g: i32, b: i32, a: i32) -> [u8; 4] {
    [
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
        a.clamp(0, 255) as u8,
    ]
}

// Adds more precision to the blue channel as described
// in c.2.14
fn blue_contract(r: i32, g: i32, b: i32, a: i32) -> [u8; 4] {
    [
        ((r + b) >> 1).clamp(0, 255) as u8,
        ((g + b) >> 1).clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
        a.clamp(0, 255) as u8,
    ]
}

// Section c.2.14
fn compute_endpoints(color_values: &mut &[u8], endpoint_mods: u32) -> [[u8; 4]; 2] {
    let ep1: [u8; 4];
    let ep2: [u8; 4];
    macro_rules! read_int_values {
        ($N:expr) => {{
            let mut v = [0; $N];
            for i in 0..$N {
                v[i] = color_values[0] as i32;
                *color_values = &color_values[1..];
            }
            v
        }};
    }

    macro_rules! bts {
        ($v:ident, $a:expr, $b: expr) => {{
            let mut a = $v[$a];
            let mut b = $v[$b];
            bit_transfer_signed(&mut a, &mut b);
            $v[$a] = a;
            $v[$b] = b;
        }};
    }

    match endpoint_mods {
        0 => {
            let v = read_int_values!(2);
            ep1 = clamp_color(v[0], v[0], v[0], 0xFF);
            ep2 = clamp_color(v[1], v[1], v[1], 0xFF);
        }

        1 => {
            let v = read_int_values!(2);
            let l0 = (v[0] >> 2) | (v[1] & 0xC0);
            let l1 = std::cmp::min(l0 + (v[1] & 0x3F), 0xFF);
            ep1 = clamp_color(l0, l0, l0, 0xFF);
            ep2 = clamp_color(l1, l1, l1, 0xFF);
        }

        4 => {
            let v = read_int_values!(4);
            ep1 = clamp_color(v[0], v[0], v[0], v[2]);
            ep2 = clamp_color(v[1], v[1], v[1], v[3]);
        }

        5 => {
            let mut v = read_int_values!(4);
            bts!(v, 1, 0);
            bts!(v, 3, 2);
            ep1 = clamp_color(v[0], v[0], v[0], v[2]);
            ep2 = clamp_color(v[0] + v[1], v[0] + v[1], v[0] + v[1], v[2] + v[3]);
        }

        6 => {
            let v = read_int_values!(4);
            ep1 = clamp_color(
                (v[0] * v[3]) >> 8,
                (v[1] * v[3]) >> 8,
                (v[2] * v[3]) >> 8,
                0xFF,
            );
            ep2 = clamp_color(v[0], v[1], v[2], 0xFF);
        }

        8 => {
            let v = read_int_values!(6);
            if v[1] + v[3] + v[5] >= v[0] + v[2] + v[4] {
                ep1 = clamp_color(v[0], v[2], v[4], 0xFF);
                ep2 = clamp_color(v[1], v[3], v[5], 0xFF);
            } else {
                ep1 = blue_contract(v[1], v[3], v[5], 0xFF);
                ep2 = blue_contract(v[0], v[2], v[4], 0xFF);
            }
        }

        9 => {
            let mut v = read_int_values!(6);
            bts!(v, 1, 0);
            bts!(v, 3, 2);
            bts!(v, 5, 4);
            if v[1] + v[3] + v[5] >= 0 {
                ep1 = clamp_color(v[0], v[2], v[4], 0xFF);
                ep2 = clamp_color(v[0] + v[1], v[2] + v[3], v[4] + v[5], 0xFF);
            } else {
                ep1 = blue_contract(v[0] + v[1], v[2] + v[3], v[4] + v[5], 0xFF);
                ep2 = blue_contract(v[0], v[2], v[4], 0xFF);
            }
        }

        10 => {
            let v = read_int_values!(6);
            ep1 = clamp_color(
                (v[0] * v[3]) >> 8,
                (v[1] * v[3]) >> 8,
                (v[2] * v[3]) >> 8,
                v[4],
            );
            ep2 = clamp_color(v[0], v[1], v[2], v[5]);
        }

        12 => {
            let v = read_int_values!(8);
            if v[1] + v[3] + v[5] >= v[0] + v[2] + v[4] {
                ep1 = clamp_color(v[0], v[2], v[4], v[6]);
                ep2 = clamp_color(v[1], v[3], v[5], v[7]);
            } else {
                ep1 = blue_contract(v[1], v[3], v[5], v[7]);
                ep2 = blue_contract(v[0], v[2], v[4], v[6]);
            }
        }

        13 => {
            let mut v = read_int_values!(8);
            bts!(v, 1, 0);
            bts!(v, 3, 2);
            bts!(v, 5, 4);
            bts!(v, 7, 6);
            if v[1] + v[3] + v[5] >= 0 {
                ep1 = clamp_color(v[0], v[2], v[4], v[6]);
                ep2 = clamp_color(v[0] + v[1], v[2] + v[3], v[4] + v[5], v[6] + v[7]);
            } else {
                ep1 = blue_contract(v[0] + v[1], v[2] + v[3], v[4] + v[5], v[6] + v[7]);
                ep2 = blue_contract(v[0], v[2], v[4], v[6]);
            }
        }

        _ => {
            // unsupported HDR modes
            ep1 = [0xFF, 0, 0xFF, 0xFF];
            ep2 = [0xFF, 0, 0xFF, 0xFF];
        }
    }
    [ep1, ep2]
}

/// Configuration for ASTC block footprint size
///
/// This struct provides predefined constants for supported footprint in the specification,
/// as well as a constructor for custom footprint. It is recommended to use only the predefined constants.
/// The constructor can be used when you want to pass in dimensions defined numerically in other file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Footprint {
    block_width: u32,
    block_height: u32,
}

impl Footprint {
    pub const ASTC_4X4: Footprint = Footprint {
        block_width: 4,
        block_height: 4,
    };
    pub const ASTC_5X4: Footprint = Footprint {
        block_width: 5,
        block_height: 4,
    };
    pub const ASTC_5X5: Footprint = Footprint {
        block_width: 5,
        block_height: 5,
    };
    pub const ASTC_6X5: Footprint = Footprint {
        block_width: 6,
        block_height: 5,
    };
    pub const ASTC_6X6: Footprint = Footprint {
        block_width: 6,
        block_height: 6,
    };
    pub const ASTC_8X5: Footprint = Footprint {
        block_width: 8,
        block_height: 5,
    };
    pub const ASTC_8X6: Footprint = Footprint {
        block_width: 8,
        block_height: 6,
    };
    pub const ASTC_10X5: Footprint = Footprint {
        block_width: 10,
        block_height: 5,
    };
    pub const ASTC_10X6: Footprint = Footprint {
        block_width: 10,
        block_height: 6,
    };
    pub const ASTC_8X8: Footprint = Footprint {
        block_width: 8,
        block_height: 8,
    };
    pub const ASTC_10X8: Footprint = Footprint {
        block_width: 10,
        block_height: 8,
    };
    pub const ASTC_10X10: Footprint = Footprint {
        block_width: 10,
        block_height: 10,
    };
    pub const ASTC_12X10: Footprint = Footprint {
        block_width: 12,
        block_height: 10,
    };
    pub const ASTC_12X12: Footprint = Footprint {
        block_width: 12,
        block_height: 12,
    };

    /// Constructs custom footprint. Panic if any of the dimensions is zero.
    pub fn new(block_width: u32, block_height: u32) -> Footprint {
        if block_width == 0 || block_height == 0 {
            panic!("Invalid block size")
        }
        Footprint {
            block_width,
            block_height,
        }
    }

    /// Returns the block width.
    pub fn block_width(&self) -> u32 {
        self.block_width
    }

    /// Returns the block height.
    pub fn block_height(&self) -> u32 {
        self.block_height
    }
}

/// Decode an ASTC block from a 16-byte buffer.
///
///  - `input` contains the raw ASTC data for one block.
///  - `footprint` specifies the footprint size.
///  - `writer` is a function with signature `FnMut(x: u32, y: u32, color: [u8; 4])`.
///     It is used to output decoded pixels.
///     This function will be called once for each pixel `(x, y)` in the rectangle
///     `[0, footprint.width()) * [0, footprint.height())`.
///     Each element in `color` represents the R, G, B, and A channel, respectively.
/// ```
/// let footprint = astc_decode::Footprint::new(6, 6);
/// // Exmaple input. Not necessarily a valid ASTC block
/// let input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// let mut output = [[0; 4]; 6 * 6];
/// astc_decode::astc_decode_block(&input, footprint, |x, y, color| {
///     output[(x + y * 6) as usize] = color;
/// });
/// ```
pub fn astc_decode_block<F: FnMut(u32, u32, [u8; 4])>(
    input: &[u8; 16],
    footprint: Footprint,
    mut writer: F,
) {
    let block_width = footprint.block_width;
    let block_height = footprint.block_height;
    let mut strm = InputBitStream::new(u128::from_le_bytes(*input));
    let weight_params = decode_block_info(&mut strm);

    // Was there an error?
    if weight_params.is_error {
        fill_error(&mut writer, block_width, block_height);
        return;
    }

    if weight_params.void_extent_ldr {
        fill_void_extent_ldr(&mut strm, &mut writer, block_width, block_height);
        return;
    }

    if weight_params.void_extent_hdr {
        fill_error(&mut writer, block_width, block_height);
        return;
    }

    if weight_params.width > block_width {
        fill_error(&mut writer, block_width, block_height);
        return;
    }

    if weight_params.height > block_height {
        fill_error(&mut writer, block_width, block_height);
        return;
    }

    // Read num partitions
    let n_partitions = (strm.read_bits(2) + 1) as usize;
    assert!(n_partitions <= 4);

    if n_partitions == 4 && weight_params.is_dual_plane {
        fill_error(&mut writer, block_width, block_height);
        return;
    }

    // Based on the number of partitions, read the color endpoint32 mode for
    // each partition.

    // Determine partitions, partition index, and color endpoint32 modes
    let plane_idx;
    let partition_index;
    let mut endpoint_mods = [0, 0, 0, 0];

    // Read extra config data...
    let mut base_cem = 0;
    if n_partitions == 1 {
        endpoint_mods[0] = strm.read_bits(4);
        partition_index = 0;
    } else {
        partition_index = strm.read_bits(10);
        base_cem = strm.read_bits(6);
    }
    let base_mode = base_cem & 3;

    // Remaining bits are color endpoint32 data...
    let n_weight_bits = weight_params.get_packed_bit_size();
    let mut remaining_bits = 128 - n_weight_bits - strm.get_bits_read();

    // Consider extra bits prior to texel data...
    let mut extra_cem_bits = 0;
    if base_mode != 0 {
        match n_partitions {
            2 => extra_cem_bits += 2,
            3 => extra_cem_bits += 5,
            4 => extra_cem_bits += 8,
            _ => panic!(),
        }
    }
    remaining_bits -= extra_cem_bits;

    // Do we have a dual plane situation?
    let mut plane_selector_bits = 0;
    if weight_params.is_dual_plane {
        plane_selector_bits = 2;
    }
    remaining_bits -= plane_selector_bits;

    // Read color data...
    let color_data_bits = remaining_bits;
    let endpoint_data = strm.read_bits128(color_data_bits);

    // Read the plane selection bits
    plane_idx = strm.read_bits(plane_selector_bits);

    // Read the rest of the cem
    if base_mode != 0 {
        let extra_cem = strm.read_bits(extra_cem_bits);
        let mut cem = (extra_cem << 6) | base_cem;
        cem >>= 2;

        let mut c = [false; 4];
        for c in &mut c[0..n_partitions] {
            *c = (cem & 1) != 0;
            cem >>= 1;
        }

        let mut m = [0; 4];
        for m in &mut m[0..n_partitions] {
            *m = cem & 3;
            cem >>= 2;
        }

        for (i, endpoint_mod) in endpoint_mods[0..n_partitions].iter_mut().enumerate() {
            *endpoint_mod = base_mode;
            if !c[i] {
                *endpoint_mod -= 1;
            }
            *endpoint_mod <<= 2;
            *endpoint_mod |= m[i];
        }
    } else if n_partitions > 1 {
        let cem = base_cem >> 2;
        endpoint_mods[0..n_partitions].fill(cem);
    }

    // Make sure everything up till here is sane.
    for &endpoint_mod in &endpoint_mods[0..n_partitions] {
        assert!(endpoint_mod < 16);
    }
    assert!(strm.get_bits_read() + weight_params.get_packed_bit_size() == 128);

    // Decode both color data and texel weight data
    let color_values =
        decode_color_values(endpoint_data, &endpoint_mods, n_partitions, color_data_bits);

    let mut endpoints = [[[0; 4]; 2]; 4];
    let mut color_values_ptr = &color_values[..];
    for i in 0..n_partitions {
        endpoints[i] = compute_endpoints(&mut color_values_ptr, endpoint_mods[i]);
    }

    // Read the texel weight data..
    let mut texel_weight_data = u128::from_le_bytes(*input).reverse_bits();

    // Make sure that higher non-texel bits are set to zero
    texel_weight_data &= (1 << weight_params.get_packed_bit_size()) - 1;

    let mut weight_stream = InputBitStream::new(texel_weight_data);

    let texel_weight_values = decode_integer_sequence(
        &mut weight_stream,
        weight_params.max_weight,
        weight_params.get_num_weight_values(),
    );

    let weights = unquantize_texel_weights(
        &texel_weight_values,
        &weight_params,
        block_width,
        block_height,
    );

    // Now that we have endpoints and weights, we can interpolate and generate
    // the proper decoding...
    for j in 0..block_height {
        for i in 0..block_width {
            let partition = select_2d_partition(
                partition_index,
                i,
                j,
                n_partitions,
                (block_height * block_width) < 32,
            );
            assert!(partition < n_partitions);

            let mut p = [0; 4];
            for (c, p) in p.iter_mut().enumerate() {
                let c0 = endpoints[partition][0][c] as u32 * 0x101;
                let c1 = endpoints[partition][1][c] as u32 * 0x101;

                let mut plane = 0;
                if weight_params.is_dual_plane && (plane_idx & 3 == c as u32) {
                    plane = 1;
                }

                let weight = weights[plane][(j * block_width + i) as usize];
                let color = (c0 * (64 - weight) + c1 * weight + 32) / 64;
                *p = u8::try_from(((color * 255) + 32767) / 65536).unwrap();
            }

            writer(i, j, p);
        }
    }
}

/// Decode an ASTC image, assuming linear block layout.
///
///  - `input` us used to read raw ASTC data.
///  - `width` and `height` specify the image dimensions.
///  - `footprint` specifies the footprint size.
///  - `writer` is a function with signature `FnMut(x: u32, y: u32, color: [u8; 4])`.
///     It is used to output decoded pixels.
///     This function will be called once for each pixel `(x, y)` in the rectangle
///     `[0, width) * [0, height)`.
///     Each element in `color` represents the R, G, B, and A channel, respectively.
///  - Returns success or IO error encountered when reading `input`.
/// ```
/// let footprint = astc_decode::Footprint::new(6, 6);
/// // Exmaple input. Not necessarily valid ASTC image
/// let input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// let mut output = [[0; 4]; 5 * 3];
/// astc_decode::astc_decode(&input[..], 5, 3, footprint, |x, y, color| {
///     output[(x + y * 5) as usize] = color;
/// }).unwrap();
/// ```
pub fn astc_decode<R: Read, F: FnMut(u32, u32, [u8; 4])>(
    mut input: R,
    width: u32,
    height: u32,
    footprint: Footprint,
    mut writer: F,
) -> Result<()> {
    let block_width = footprint.block_width;
    let block_height = footprint.block_height;

    let block_w = (width.checked_add(block_width).unwrap() - 1) / block_width;
    let block_h = (height.checked_add(block_height).unwrap() - 1) / block_height;

    for by in 0..block_h {
        for bx in 0..block_w {
            let mut block_buf = [0; 16];
            input.read_exact(&mut block_buf)?;
            astc_decode_block(&block_buf, footprint, |x, y, v| {
                let x = bx * block_width + x;
                let y = by * block_height + y;
                if x < width && y < height {
                    writer(x, y, v)
                }
            })
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Pixel;

    fn dist(a: u8, b: u8) {
        assert!((a as i32 - b as i32).abs() <= 1)
    }

    fn test_case(astc: &[u8], bmp: &[u8], block_width: u32, block_height: u32) {
        let bmp = image::load_from_memory(bmp).unwrap().to_rgba8();
        let width = bmp.width();
        let height = bmp.height();
        astc_decode(
            &astc[16..],
            width,
            height,
            Footprint::new(block_width, block_height),
            |x, y, v| {
                let y = height - y - 1;
                let p = bmp.get_pixel(x as u32, y as u32).channels();
                dist(p[0], v[0]);
                dist(p[1], v[1]);
                dist(p[2], v[2]);
                dist(p[3], v[3]);
            },
        )
        .unwrap();
    }

    macro_rules! tc {
        ($name:literal, $bw:literal, $bh:literal) => {
            test_case(
                include_bytes!(concat!("test-data/", $name, '_', $bw, 'x', $bh, ".astc")),
                include_bytes!(concat!("test-data/", $name, '_', $bw, 'x', $bh, ".bmp")),
                $bw,
                $bh,
            );
        };
    }

    #[test]
    fn it_works() {
        tc!("atlas_small", 4, 4);
        tc!("atlas_small", 5, 5);
        tc!("atlas_small", 6, 6);
        tc!("atlas_small", 8, 8);
        tc!("footprint", 4, 4);
        tc!("footprint", 5, 4);
        tc!("footprint", 5, 5);
        tc!("footprint", 6, 5);
        tc!("footprint", 6, 6);
        tc!("footprint", 8, 5);
        tc!("footprint", 8, 6);
        tc!("footprint", 8, 8);
        tc!("footprint", 10, 5);
        tc!("footprint", 10, 6);
        tc!("footprint", 10, 8);
        tc!("footprint", 10, 10);
        tc!("footprint", 12, 10);
        tc!("footprint", 12, 12);
        tc!("rgb", 4, 4);
        tc!("rgb", 5, 4);
        tc!("rgb", 6, 6);
        tc!("rgb", 8, 8);
        tc!("rgb", 12, 12);
    }
}

#![forbid(unsafe_code)]
//! Auto-generated & verified by `a6000-orchestrator` Subsystem F on `49.13.101.247` (RTX 6000 Ada).
//! Repository: `Engram` · Target Symbol: `triad_engram_zero_float_step` · Kind: `triad_3_agent_zero_float_synthesis`
//! Strict `8b.is` No-Float Law: 100% integer & balanced ternary (`{-1, 0, +1}`) arithmetic.

pub const K_CONST: u64 = 30665;

#[inline(never)]
pub fn baseline_lane_step(word: u64, lane: u64, k: u64) -> i64 {
let shifted = word.rotate_left((lane as u32) * 4) ^ k.wrapping_mul(lane + 1);
let rem = shifted % 3;
let trit: i64 = if rem == 0 {
0
} else if rem == 1 {
1
} else {
-1
};
let folded = (shifted >> 3).wrapping_add((trit as u64).wrapping_mul(432)) as i64;
std::hint::black_box(folded ^ (trit * 528))
}

#[inline(never)]
pub fn baseline_kernel(seed: u64) -> i64 {
let mut acc: i64 = 0;
let mut lane: u64 = 0;
while lane < 16 {
acc = acc.wrapping_add(baseline_lane_step(seed, lane, K_CONST));
lane += 1;
}
acc
}

#[inline(always)]
pub const fn improved_lane_fast(word: u64, lane: u64, k: u64) -> i64 {
let shifted = word.rotate_left((lane as u32) * 4) ^ k.wrapping_mul(lane + 1);
let rem = shifted % 3;
let trit: i64 = (rem == 1) as i64 - (rem == 2) as i64;
let folded = (shifted >> 3).wrapping_add((trit as u64).wrapping_mul(432)) as i64;
folded ^ (trit * 528)
}

#[inline(never)]
pub fn improved_kernel(seed: u64) -> i64 {
let a0 = improved_lane_fast(seed, 0, K_CONST)
.wrapping_add(improved_lane_fast(seed, 1, K_CONST))
.wrapping_add(improved_lane_fast(seed, 2, K_CONST))
.wrapping_add(improved_lane_fast(seed, 3, K_CONST));
let a1 = improved_lane_fast(seed, 4, K_CONST)
.wrapping_add(improved_lane_fast(seed, 5, K_CONST))
.wrapping_add(improved_lane_fast(seed, 6, K_CONST))
.wrapping_add(improved_lane_fast(seed, 7, K_CONST));
let a2 = improved_lane_fast(seed, 8, K_CONST)
.wrapping_add(improved_lane_fast(seed, 9, K_CONST))
.wrapping_add(improved_lane_fast(seed, 10, K_CONST))
.wrapping_add(improved_lane_fast(seed, 11, K_CONST));
let a3 = improved_lane_fast(seed, 12, K_CONST)
.wrapping_add(improved_lane_fast(seed, 13, K_CONST))
.wrapping_add(improved_lane_fast(seed, 14, K_CONST))
.wrapping_add(improved_lane_fast(seed, 15, K_CONST));
a0.wrapping_add(a1).wrapping_add(a2).wrapping_add(a3)
}

#[cfg(test)]
mod tests {
use super::*;

#[test]
fn test_a6000_autonomous_64_vector_equivalence() {
for idx in 0..64u64 {
let vec_seed = idx.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 432;
assert_eq!(baseline_kernel(vec_seed), improved_kernel(vec_seed));
}
}

#[test]
fn test_a6000_branchless_trit_stability() {
let v0 = improved_kernel(0);
let v432 = improved_kernel(432);
assert_eq!(v0, baseline_kernel(0));
assert_eq!(v432, baseline_kernel(432));
}
}

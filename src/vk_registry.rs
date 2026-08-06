//! Stateless prepared-operand syscalls for registry-backed verification.
//!
//! The runtime validates only encodings; provenance of prepared blobs is the
//! calling program's obligation. A program stores blobs produced by
//! [`g2_prepare`] in its own account, authenticates that account itself
//! (address commitment), and passes the borrowed bytes here. These syscalls
//! exist only on the `local/bn254-prepared-stateless` agave fork; the
//! declarations are vendored until a published crate ships them, pinned by
//! the layout tests below.

use crate::errors::Groth16Error;

pub const PREPARED_ABI_VERSION: u16 = 1;
pub const PREPARED_BLOB_HEADER: [u8; 8] = *b"BPG2\x01\x01\x05\x00";
/// 8-byte self-describing header plus 87 line triples x 3 Fp2 x 2 Fp x 4 u64
/// little-endian Montgomery limbs.
pub const PREPARED_G2_WIRE_BYTES: usize = 8 + 87 * 3 * 2 * 4 * 8;
pub const MAX_PREPARED_PAIRS: usize = 16;
pub const GT_BYTES: usize = 384;

/// VM-address reference to one prepared wire blob; alignment 1 so a record
/// can point at any account-data offset.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PreparedRef {
    pub addr_le: [u8; 8],
    pub len_le: [u8; 8],
}

/// One pairing operand with a caller-supplied prepared G2, 80 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct G1PreparedPair {
    pub g1: [u8; 64],
    pub prepared: PreparedRef,
}

/// One full pairing operand, 192 bytes, both points validated in-syscall.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct G1G2Pair {
    pub g1: [u8; 64],
    pub g2: [u8; 128],
}

const _: () = {
    assert!(core::mem::size_of::<PreparedRef>() == 16);
    assert!(core::mem::align_of::<PreparedRef>() == 1);
    assert!(core::mem::size_of::<G1PreparedPair>() == 80);
    assert!(core::mem::align_of::<G1PreparedPair>() == 1);
    assert!(core::mem::size_of::<G1G2Pair>() == 192);
    assert!(core::mem::align_of::<G1G2Pair>() == 1);
    assert!(PREPARED_G2_WIRE_BYTES == 16_712);
};

impl G1PreparedPair {
    /// Reference `blob` in place; the runtime translates and validates it.
    /// `blob` must stay borrowed for the duration of the syscall.
    pub fn new(g1: [u8; 64], blob: &[u8]) -> Self {
        Self {
            g1,
            prepared: PreparedRef {
                addr_le: (blob.as_ptr() as u64).to_le_bytes(),
                len_le: (blob.len() as u64).to_le_bytes(),
            },
        }
    }
}

pub const fn pack_g2_prepare_shape() -> u64 {
    (PREPARED_ABI_VERSION as u64) << 48
}

/// Bits 0..16 full count, 16..32 prepared count, 32..48 reserved zero,
/// 48..64 ABI version.
pub const fn pack_prepared_pairing_shape(full_count: u16, prepared_count: u16) -> u64 {
    full_count as u64 | ((prepared_count as u64) << 16) | ((PREPARED_ABI_VERSION as u64) << 48)
}

#[cfg(target_os = "solana")]
mod inner {
    use super::*;

    solana_define_syscall::define_syscall!(fn sol_alt_bn128_pairing_map(num_pairs: u64, pairs_addr: *const u8, result_addr: *mut u8) -> u64);
    solana_define_syscall::define_syscall!(fn sol_alt_bn128_g2_prepare(packed_shape: u64, g2_source_addr: *const u8, prepared_out_addr: *mut u8) -> u64);
    solana_define_syscall::define_syscall!(fn sol_alt_bn128_pairing_check_prepared(packed_shape: u64, full_addr: *const u8, prepared_addr: *const u8, target_addr: *const u8, result_addr: *mut u8) -> u64);
    solana_define_syscall::define_syscall!(fn sol_alt_bn128_pairing_map_prepared(packed_shape: u64, full_addr: *const u8, prepared_addr: *const u8, result_addr: *mut u8) -> u64);

    /// Fully validate one canonical G2 point and write its prepared wire
    /// blob into `out`. Registration-time only.
    pub fn g2_prepare(source: &[u8; 128], out: &mut [u8]) -> Result<(), Groth16Error> {
        if out.len() != PREPARED_G2_WIRE_BYTES {
            return Err(Groth16Error::VkRegistrySyscallFailed);
        }
        let rc = unsafe {
            sol_alt_bn128_g2_prepare(pack_g2_prepare_shape(), source.as_ptr(), out.as_mut_ptr())
        };
        if rc != 0 {
            return Err(Groth16Error::VkRegistrySyscallFailed);
        }
        Ok(())
    }

    /// Post-final-exponentiation product of fully validated pairs; computes
    /// the cached GT target at registration time.
    pub fn pairing_map(pairs: &[G1G2Pair]) -> Result<[u8; GT_BYTES], Groth16Error> {
        let mut result = [0u8; GT_BYTES];
        let rc = unsafe {
            sol_alt_bn128_pairing_map(
                pairs.len() as u64,
                pairs.as_ptr().cast(),
                result.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(Groth16Error::VkRegistrySyscallFailed);
        }
        Ok(result)
    }

    /// Mixed product verdict. `target` of `None` compares against the GT
    /// identity; `Some` compares against bytes previously returned by
    /// [`pairing_map`]. Prepared operands skip the subgroup check and line
    /// preparation; only the caller's own verification is at stake.
    pub fn pairing_check_prepared(
        full: &[G1G2Pair],
        prepared: &[G1PreparedPair],
        target: Option<&[u8; GT_BYTES]>,
    ) -> Result<bool, Groth16Error> {
        let total = full
            .len()
            .checked_add(prepared.len())
            .ok_or(Groth16Error::VkRegistrySyscallFailed)?;
        if total == 0 || prepared.len() > MAX_PREPARED_PAIRS {
            return Err(Groth16Error::VkRegistrySyscallFailed);
        }
        let shape = pack_prepared_pairing_shape(full.len() as u16, prepared.len() as u16);
        let mut result = [0u8; 32];
        let fallback = result.as_ptr();
        let full_addr = if full.is_empty() {
            fallback
        } else {
            full.as_ptr().cast()
        };
        let prepared_addr = if prepared.is_empty() {
            fallback
        } else {
            prepared.as_ptr().cast()
        };
        let target_addr = target.map_or(core::ptr::null(), |target| target.as_ptr());
        let rc = unsafe {
            sol_alt_bn128_pairing_check_prepared(
                shape,
                full_addr,
                prepared_addr,
                target_addr,
                result.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(Groth16Error::VkRegistrySyscallFailed);
        }
        let mut expected = [0u8; 32];
        expected[31] = 1;
        Ok(result == expected)
    }

    /// Mixed product mapped to its canonical 384-byte encoding.
    pub fn pairing_map_prepared(
        full: &[G1G2Pair],
        prepared: &[G1PreparedPair],
    ) -> Result<[u8; GT_BYTES], Groth16Error> {
        let total = full
            .len()
            .checked_add(prepared.len())
            .ok_or(Groth16Error::VkRegistrySyscallFailed)?;
        if total == 0 || prepared.len() > MAX_PREPARED_PAIRS {
            return Err(Groth16Error::VkRegistrySyscallFailed);
        }
        let shape = pack_prepared_pairing_shape(full.len() as u16, prepared.len() as u16);
        let mut result = [0u8; GT_BYTES];
        let fallback = result.as_ptr();
        let full_addr = if full.is_empty() {
            fallback
        } else {
            full.as_ptr().cast()
        };
        let prepared_addr = if prepared.is_empty() {
            fallback
        } else {
            prepared.as_ptr().cast()
        };
        let rc = unsafe {
            sol_alt_bn128_pairing_map_prepared(shape, full_addr, prepared_addr, result.as_mut_ptr())
        };
        if rc != 0 {
            return Err(Groth16Error::VkRegistrySyscallFailed);
        }
        Ok(result)
    }
}

// The prepared syscalls have no host equivalent in this crate; host-side
// coverage runs through the fork's litesvm harness in the consumer repo.
#[cfg(not(target_os = "solana"))]
mod inner {
    use super::*;

    pub fn g2_prepare(_source: &[u8; 128], _out: &mut [u8]) -> Result<(), Groth16Error> {
        Err(Groth16Error::VkRegistryUnsupportedHost)
    }

    pub fn pairing_map(_pairs: &[G1G2Pair]) -> Result<[u8; GT_BYTES], Groth16Error> {
        Err(Groth16Error::VkRegistryUnsupportedHost)
    }

    pub fn pairing_check_prepared(
        _full: &[G1G2Pair],
        _prepared: &[G1PreparedPair],
        _target: Option<&[u8; GT_BYTES]>,
    ) -> Result<bool, Groth16Error> {
        Err(Groth16Error::VkRegistryUnsupportedHost)
    }

    pub fn pairing_map_prepared(
        _full: &[G1G2Pair],
        _prepared: &[G1PreparedPair],
    ) -> Result<[u8; GT_BYTES], Groth16Error> {
        Err(Groth16Error::VkRegistryUnsupportedHost)
    }
}

pub use inner::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_packing_is_pinned() {
        assert_eq!(pack_g2_prepare_shape(), 1 << 48);
        assert_eq!(pack_prepared_pairing_shape(5, 3), 5 | (3 << 16) | (1 << 48));
        assert_eq!(PREPARED_BLOB_HEADER, [66, 80, 71, 50, 1, 1, 5, 0]);
    }

    #[test]
    fn prepared_pair_references_the_blob_in_place() {
        let blob = [0u8; 4];
        let pair = G1PreparedPair::new([7u8; 64], &blob);
        assert_eq!(
            u64::from_le_bytes(pair.prepared.addr_le),
            blob.as_ptr() as u64
        );
        assert_eq!(u64::from_le_bytes(pair.prepared.len_le), 4);
    }
}

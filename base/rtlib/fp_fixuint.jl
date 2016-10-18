#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#
#
# This file implements float to unsigned integer conversion for the
# compiler-rt library.
#

@inline function fixuint{fixuint_t<:Unsigned, fp_t<:RTLIB_FLOAT}(::Type{fixint_t}, a::fp_t)
    const rep_t = fptoui(fp_t)
    # Get masks
    const signBit = one(rep_t) << (significand_bits(fp_t) + exponent_bits(fp_t))
    const absMask = signBit - one(rep_t)
    const implicitBit = one(rep_t) << significand_bits(fp_t)

    # Break a into sign, exponent, significand
    const aRep = reinterpret(rep_t, a)
    const aAbs = aRep & absMask
    const sign = ifelse(aRep & signBit != 0, -1, 1)
    const exponent :: rep_t = (aAbs >> significand_bits(fp_t)) - exponent_bias(fp_t)
    const significand :: rep_t = (aAbs & significand_mask(fp_t)) | implicitBit;

    # If either the value or the exponent is negative, the result is zero.
    if (sign == -1 || exponent < 0)
        return zero(fixuint_t);
    end

    # If the value is too large for the integer type, saturate.
    if exponent >= nbits(fixuint_t)
        return typemax(fixuint_t);
    end

    # If 0 <= exponent < significandBits, right shift to get the result.
    # Otherwise, shift left.
    if exponent < significand_bits(fp_t)
        return (significand >> (significand_bits(fp_t) - exponent)) % fixuint_t
    else
        return (significand % fixuint_t) << (exponent - significand_bits(fp_t))
    end
end

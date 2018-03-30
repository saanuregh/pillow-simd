#include "Imaging.h"
#include <math.h>

#include <emmintrin.h>
#include <mmintrin.h>
#include <smmintrin.h>
#if defined(__AVX2__)
    #include <immintrin.h>
#endif


/* 8 bits for result. Table can overflow [0, 1.0] range,
   so we need extra bits for overflow and negative values.
   NOTE: This value should be the same as in _imaging/_prepare_lut_table() */
#define PRECISION_BITS (16 - 8 - 2)
#define PRECISION_ROUNDING (1<<(PRECISION_BITS-1))

/* 8 — scales are multiplied on byte.
   6 — max index in the table
       (max size is 65, but index 64 is not reachable) */
#define SCALE_BITS (32 - 8 - 6)
#define SCALE_MASK ((1<<SCALE_BITS) - 1)

#define SHIFT_BITS (16 - 1)


/*
 Transforms colors of imIn using provided 3D lookup table
 and puts the result in imOut. Returns imOut on sucess or 0 on error.
 
 imOut, imIn — images, should be the same size and may be the same image.
    Should have 3 or 4 channels.
 table_channels — number of channels in the lookup table, 3 or 4.
    Should be less or equal than number of channels in imOut image;
 size1D, size_2D and size3D — dimensions of provided table;
 table — flat table,
    array with table_channels × size1D × size2D × size3D elements,
    where channels are changed first, then 1D, then​ 2D, then 3D.
    Each element is signed 16-bit int where 0 is lowest output value
    and 255 << PRECISION_BITS (16320) is highest value.
*/ 
Imaging
ImagingColorLUT3D_linear(Imaging imOut, Imaging imIn, int table_channels,
                         int size1D, int size2D, int size3D,
                         INT16* table)
{
    /* This float to int conversion doesn't have rounding
       error compensation (+0.5) for two reasons:
       1. As we don't hit the highest value,
          we can use one extra bit for precision.
       2. For every pixel, we interpolate 8 elements from the table:
          current and +1 for every dimension and their combinations.
          If we hit the upper cells from the table,
          +1 cells will be outside of the table.
          With this compensation we never hit the upper cells
          but this also doesn't introduce any noticeable difference. */
    int y, size1D_2D = size1D * size2D;
#if defined(__AVX2__)
    __m256i scale256 = _mm256_set_epi32(
        0,
        (size3D - 1) / 255.0 * (1<<SCALE_BITS),
        (size2D - 1) / 255.0 * (1<<SCALE_BITS),
        (size1D - 1) / 255.0 * (1<<SCALE_BITS),
        0,
        (size3D - 1) / 255.0 * (1<<SCALE_BITS),
        (size2D - 1) / 255.0 * (1<<SCALE_BITS),
        (size1D - 1) / 255.0 * (1<<SCALE_BITS));
    __m256i index_mul256 = _mm256_set_epi32(
        0, size1D_2D*table_channels, size1D*table_channels, table_channels,
        0, size1D_2D*table_channels, size1D*table_channels, table_channels);
#endif

    if (table_channels < 3 || table_channels > 4) {
        PyErr_SetString(PyExc_ValueError, "table_channels could be 3 or 4");
        return NULL;
    }

    if (imIn->type != IMAGING_TYPE_UINT8 ||
        imOut->type != IMAGING_TYPE_UINT8 ||
        imIn->bands < 3 ||
        imOut->bands < table_channels
    ) {
        return (Imaging) ImagingError_ModeError();
    }

    /* In case we have one extra band in imOut and don't have in imIn.*/
    if (imOut->bands > table_channels && imOut->bands > imIn->bands) {
        return (Imaging) ImagingError_ModeError();
    }

    // {
    //     int unalignIn = 0;
    //     int unalignOut = 0;
    //     for (y = 0; y < imIn->ysize; y++) {
    //         if (0x0f & (uintptr_t) imIn->image[y]) {
    //             unalignIn += 1;
    //         }
    //         if (0x0f & (uintptr_t) imOut->image[y]) {
    //             unalignOut += 1;
    //         }
    //     }
    //     printf("unalignIn: %d, unalignOut: %d, table: %p\n",
    //            unalignIn, unalignOut, table);
    // }

    for (y = 0; y < imOut->ysize; y++) {
        UINT32* rowIn = (UINT32 *)imIn->image[y];
        UINT32* rowOut = (UINT32 *)imOut->image[y];
        int x = 0;

    #if defined(__AVX2__)
    {
        __m256i index = _mm256_mullo_epi32(scale256,
            _mm256_cvtepu8_epi32(*(__m128i *) &rowIn[x]));
        __m256i idxs = _mm256_hadd_epi32(_mm256_hadd_epi32(
            _mm256_madd_epi16(index_mul256, _mm256_srli_epi32(index, SCALE_BITS)),
            _mm256_setzero_si256()), _mm256_setzero_si256());
        int idx1 = _mm_cvtsi128_si32(_mm256_castsi256_si128(idxs));
        int idx2 = _mm256_extract_epi32(idxs, 4);

        for (; x < imOut->xsize - 4; x += 2) {
            __m256i next_index = _mm256_mullo_epi32(scale256,
                _mm256_cvtepu8_epi32(*(__m128i *) &rowIn[x + 2]));
            __m256i next_idxs = _mm256_hadd_epi32(_mm256_hadd_epi32(
                _mm256_madd_epi16(index_mul256, _mm256_srli_epi32(next_index, SCALE_BITS)),
                _mm256_setzero_si256()), _mm256_setzero_si256());
            int next_idx1 = _mm_cvtsi128_si32(_mm256_castsi256_si128(next_idxs));
            int next_idx2 = _mm256_extract_epi32(next_idxs, 4);

            __m256i result;

            result = _mm256_inserti128_si256(_mm256_castsi128_si256(
                    _mm_loadu_si128((__m128i *) &table[idx1 + 0])),
                    _mm_loadu_si128((__m128i *) &table[idx2 + 0]), 1);

            result = _mm256_srai_epi32(_mm256_add_epi32(
                _mm256_set1_epi32(PRECISION_ROUNDING), result),
                PRECISION_BITS);

            result = _mm256_packs_epi32(result, result);
            result = _mm256_packus_epi16(result, result);
            rowOut[x + 0] = _mm_cvtsi128_si32(_mm256_castsi256_si128(result));
            rowOut[x + 1] = _mm256_extract_epi32(result, 4);

            idx1 = next_idx1;
            idx2 = next_idx2;
        }
    }
    #endif

    }

    return imOut;
}

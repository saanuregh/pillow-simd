#include "Python.h"
#include "Imaging.h"

#include <emmintrin.h>
#include <mmintrin.h>
#include <smmintrin.h>

#if defined(__AVX2__)
    #include <immintrin.h>
#endif


#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* Number of bits in 32-bit accumulator not occupied by the final 8-bit value */
#define ACC_MAX_PRECISION 24

typedef UINT8 pixel[4];


int static
compute_params(float fRadius, int maxSize,
               int *edgeA, int *edgeB, UINT32 *ww, UINT32 *fw)
{
    float error;
    int radius = (int) fRadius;

    /* The hypothetical error which could be
       if we round the radius to the nearest integer. */
    error = fabs(fRadius - round(fRadius)) * 2 / (round(fRadius) * 2 + 1);

    // printf(">>> %f (%f, ", fRadius, error);

    /* If the error is small enough, use simpler implementation.
       Works for near-integer radii and large radii. */
    if (256 * error < 1) {
        radius = fRadius = round(fRadius);
    }

    *ww = (UINT32) (1 << 18) / (fRadius * 2 + 1);
    *fw = (UINT32) (1 << 18) * (fRadius - radius) / (fRadius * 2 + 1);

    *edgeA = MIN(radius + 1, maxSize);
    *edgeB = MAX(maxSize - radius - 1, 0);

    // printf("%d), from %d to %d. %d %d\n", radius, *edgeA, *edgeB, *ww, *fw);

    return radius;
}


/* General implementation when radius > 0 and not too big */
void static inline
ImagingLineBoxBlur4(pixel *lineOut, pixel *lineIn, int lastx, int radius,
                    int edgeA, int edgeB, UINT32 ww, UINT32 fw)
{
    int x;
    UINT32 acc[4];
    UINT32 bulk[4];

    #define MOVE_ACC(acc, subtract, add) \
        acc[0] += lineIn[add][0] - lineIn[subtract][0]; \
        acc[1] += lineIn[add][1] - lineIn[subtract][1]; \
        acc[2] += lineIn[add][2] - lineIn[subtract][2]; \
        acc[3] += lineIn[add][3] - lineIn[subtract][3];

    #define ADD_FAR(bulk, acc, left, right) \
        bulk[0] = (acc[0] * ww) + (lineIn[left][0] + lineIn[right][0]) * fw; \
        bulk[1] = (acc[1] * ww) + (lineIn[left][1] + lineIn[right][1]) * fw; \
        bulk[2] = (acc[2] * ww) + (lineIn[left][2] + lineIn[right][2]) * fw; \
        bulk[3] = (acc[3] * ww) + (lineIn[left][3] + lineIn[right][3]) * fw;

    #define SAVE(x, bulk) \
        lineOut[x][0] = (UINT8)((bulk[0] + (1 << 23)) >> 24); \
        lineOut[x][1] = (UINT8)((bulk[1] + (1 << 23)) >> 24); \
        lineOut[x][2] = (UINT8)((bulk[2] + (1 << 23)) >> 24); \
        lineOut[x][3] = (UINT8)((bulk[3] + (1 << 23)) >> 24);

    /* Compute acc for -1 pixel (outside of image):
       From "-radius-1" to "-1" get first pixel,
       then from "0" to "radius-1". */
    acc[0] = lineIn[0][0] * (radius + 1);
    acc[1] = lineIn[0][1] * (radius + 1);
    acc[2] = lineIn[0][2] * (radius + 1);
    acc[3] = lineIn[0][3] * (radius + 1);
    /* As radius can be bigger than xsize, iterate to edgeA -1. */
    for (x = 0; x < edgeA - 1; x++) {
        acc[0] += lineIn[x][0];
        acc[1] += lineIn[x][1];
        acc[2] += lineIn[x][2];
        acc[3] += lineIn[x][3];
    }
    /* Then multiply remainder to last x. */
    acc[0] += lineIn[lastx][0] * (radius - edgeA + 1);
    acc[1] += lineIn[lastx][1] * (radius - edgeA + 1);
    acc[2] += lineIn[lastx][2] * (radius - edgeA + 1);
    acc[3] += lineIn[lastx][3] * (radius - edgeA + 1);

    if (edgeA <= edgeB)
    {
        /* Subtract pixel from left ("0").
           Add pixels from radius. */
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        /* Subtract previous pixel from "-radius".
           Add pixels from radius. */
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        /* Subtract previous pixel from "-radius".
           Add last pixel. */
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }
    else
    {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

    #undef MOVE_ACC
    #undef ADD_FAR
    #undef SAVE
}


/* Implementation for large or integer radii */
void static inline
ImagingLineBoxBlur4Large(pixel *lineOut, pixel *lineIn, int lastx, int radius,
                         int edgeA, int edgeB, UINT32 ww)
{
    int x;
    UINT32 acc[4];
    UINT32 bulk[4];

    #define MOVE_ACC(acc, subtract, add) \
        acc[0] += lineIn[add][0] - lineIn[subtract][0]; \
        acc[1] += lineIn[add][1] - lineIn[subtract][1]; \
        acc[2] += lineIn[add][2] - lineIn[subtract][2]; \
        acc[3] += lineIn[add][3] - lineIn[subtract][3];

    #define ADD_FAR(bulk, acc) \
        bulk[0] = acc[0] * ww; \
        bulk[1] = acc[1] * ww; \
        bulk[2] = acc[2] * ww; \
        bulk[3] = acc[3] * ww;

    #define SAVE(x, bulk) \
        lineOut[x][0] = (UINT8)((bulk[0] + (1 << 23)) >> 24); \
        lineOut[x][1] = (UINT8)((bulk[1] + (1 << 23)) >> 24); \
        lineOut[x][2] = (UINT8)((bulk[2] + (1 << 23)) >> 24); \
        lineOut[x][3] = (UINT8)((bulk[3] + (1 << 23)) >> 24);

    /* Compute acc for -1 pixel (outside of image):
       From "-radius-1" to "-1" get first pixel,
       then from "0" to "radius-1". */
    acc[0] = lineIn[0][0] * (radius + 1);
    acc[1] = lineIn[0][1] * (radius + 1);
    acc[2] = lineIn[0][2] * (radius + 1);
    acc[3] = lineIn[0][3] * (radius + 1);
    /* As radius can be bigger than xsize, iterate to edgeA -1. */
    for (x = 0; x < edgeA - 1; x++) {
        acc[0] += lineIn[x][0];
        acc[1] += lineIn[x][1];
        acc[2] += lineIn[x][2];
        acc[3] += lineIn[x][3];
    }
    /* Then multiply remainder to last x. */
    acc[0] += lineIn[lastx][0] * (radius - edgeA + 1);
    acc[1] += lineIn[lastx][1] * (radius - edgeA + 1);
    acc[2] += lineIn[lastx][2] * (radius - edgeA + 1);
    acc[3] += lineIn[lastx][3] * (radius - edgeA + 1);

    if (edgeA <= edgeB)
    {
        /* Subtract pixel from left ("0").
           Add pixels from radius. */
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
        /* Subtract previous pixel from "-radius".
           Add pixels from radius. */
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
        /* Subtract previous pixel from "-radius".
           Add last pixel. */
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
    }
    else
    {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
    }

    #undef MOVE_ACC
    #undef ADD_FAR
    #undef SAVE
}


/* Optimized implementation when radius = 0 */
void static inline
ImagingLineBoxBlur4Zero(pixel *lineOut, pixel *lineIn, int lastx,
    int edgeA, int edgeB, UINT32 ww, UINT32 fw)
{
    int x;
    UINT32 acc[4];
    UINT32 bulk[4];

    #define MOVE_ACC(acc, add) \
        acc[0] = lineIn[add][0]; \
        acc[1] = lineIn[add][1]; \
        acc[2] = lineIn[add][2]; \
        acc[3] = lineIn[add][3];

    #define ADD_FAR(bulk, acc, left, right) \
        bulk[0] = (acc[0] * ww) + (lineIn[left][0] + lineIn[right][0]) * fw; \
        bulk[1] = (acc[1] * ww) + (lineIn[left][1] + lineIn[right][1]) * fw; \
        bulk[2] = (acc[2] * ww) + (lineIn[left][2] + lineIn[right][2]) * fw; \
        bulk[3] = (acc[3] * ww) + (lineIn[left][3] + lineIn[right][3]) * fw;

    #define SAVE(x, bulk) \
        lineOut[x][0] = (UINT8)((bulk[0] + (1 << 23)) >> 24); \
        lineOut[x][1] = (UINT8)((bulk[1] + (1 << 23)) >> 24); \
        lineOut[x][2] = (UINT8)((bulk[2] + (1 << 23)) >> 24); \
        lineOut[x][3] = (UINT8)((bulk[3] + (1 << 23)) >> 24);

    if (edgeA <= edgeB)
    {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, x);
            ADD_FAR(bulk, acc, 0, x + 1);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x);
            ADD_FAR(bulk, acc, x - 1, x + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, lastx);
            ADD_FAR(bulk, acc, x - 1, lastx);
            SAVE(x, bulk);
        }
    }
    else
    {
        /* This is possible when radius = 0 and width = 1. */
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
    }

    #undef MOVE_ACC
    #undef ADD_FAR
    #undef SAVE
}


/* General implementation when radius > 0 and not too big */
void static inline
ImagingLineBoxBlur1(UINT8 *lineOut, UINT8 *lineIn, int lastx, int radius,
                    int edgeA, int edgeB, UINT32 ww, UINT32 fw)
{
    int x;
    UINT32 acc;
    UINT32 bulk;

    #define MOVE_ACC(acc, subtract, add) \
        acc += lineIn[add] - lineIn[subtract];

    #define ADD_FAR(bulk, acc, left, right) \
        bulk = (acc * ww) + (lineIn[left] + lineIn[right]) * fw;

    #define SAVE(x, bulk) \
        lineOut[x] = (UINT8)((bulk + (1 << 23)) >> 24)

    acc = lineIn[0] * (radius + 1);
    for (x = 0; x < edgeA - 1; x++) {
        acc += lineIn[x];
    }
    acc += lineIn[lastx] * (radius - edgeA + 1);

    if (edgeA <= edgeB)
    {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }
    else
    {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

    #undef MOVE_ACC
    #undef ADD_FAR
    #undef SAVE
}


/* Implementation for large or integer radii */
void static inline
ImagingLineBoxBlur1Large(UINT8 *lineOut, UINT8 *lineIn, int lastx, int radius,
                         int edgeA, int edgeB, UINT32 ww)
{
    int x;
    UINT32 acc;
    UINT32 bulk;

    #define MOVE_ACC(acc, subtract, add) \
        acc += lineIn[add] - lineIn[subtract];

    #define ADD_FAR(bulk, acc) \
        bulk = acc * ww;

    #define SAVE(x, bulk) \
        lineOut[x] = (UINT8)((bulk + (1 << 23)) >> 24)

    acc = lineIn[0] * (radius + 1);
    for (x = 0; x < edgeA - 1; x++) {
        acc += lineIn[x];
    }
    acc += lineIn[lastx] * (radius - edgeA + 1);

    if (edgeA <= edgeB)
    {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
    }
    else
    {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc);
            SAVE(x, bulk);
        }
    }

    #undef MOVE_ACC
    #undef ADD_FAR
    #undef SAVE
}


/* Optimized implementation when radius = 0 */
void static inline
ImagingLineBoxBlur1Zero(UINT8 *lineOut, UINT8 *lineIn, int lastx,
                        int edgeA, int edgeB, UINT32 ww, UINT32 fw)
{
    int x;
    UINT32 acc;
    UINT32 bulk;

    #define MOVE_ACC(acc, add) \
        acc = lineIn[add];

    #define ADD_FAR(bulk, acc, left, right) \
        bulk = (acc * ww) + (lineIn[left] + lineIn[right]) * fw;

    #define SAVE(x, bulk) \
        lineOut[x] = (UINT8)((bulk + (1 << 23)) >> 24)

    if (edgeA <= edgeB) {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, x);
            ADD_FAR(bulk, acc, 0, x + 1);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x);
            ADD_FAR(bulk, acc, x - 1, x + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, lastx);
            ADD_FAR(bulk, acc, x - 1, lastx);
            SAVE(x, bulk);
        }
    } else {
        /* This is possible when radius = 0 and width = 1. */
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
    }

    #undef MOVE_ACC
    #undef ADD_FAR
    #undef SAVE
}


Imaging
ImagingHorizontalBoxBlur(Imaging imOut, Imaging imIn, float fRadius)
{
    ImagingSectionCookie cookie;

    UINT32 ww, fw;
    int edgeA, edgeB;
    int radius = compute_params(fRadius, imIn->xsize, &edgeA, &edgeB, &ww, &fw);
    int y;

    ImagingSectionEnter(&cookie);

    if (imIn->image8)
    {
        for (y = 0; y < imIn->ysize; y++) {
            if (radius == 0) {
                ImagingLineBoxBlur1Zero(
                    imOut->image8[y], imIn->image8[y],
                    imIn->xsize - 1, edgeA, edgeB, ww, fw);
            } else if (fw == 0) {
                ImagingLineBoxBlur1Large(
                    imOut->image8[y], imIn->image8[y],
                    imIn->xsize - 1, radius, edgeA, edgeB, ww);
            } else {
                ImagingLineBoxBlur1(
                    imOut->image8[y], imIn->image8[y],
                    imIn->xsize - 1, radius, edgeA, edgeB, ww, fw);
            }
        }
    }
    else
    {
        for (y = 0; y < imIn->ysize; y++) {
            if (radius == 0) {
                ImagingLineBoxBlur4Zero(
                    (pixel *) imOut->image32[y], (pixel *) imIn->image32[y],
                    imIn->xsize - 1, edgeA, edgeB, ww, fw);
            } else if (fw == 0) {
                ImagingLineBoxBlur4Large(
                    (pixel *) imOut->image32[y], (pixel *) imIn->image32[y],
                    imIn->xsize - 1, radius, edgeA, edgeB, ww);
            } else {
                ImagingLineBoxBlur4(
                    (pixel *) imOut->image32[y], (pixel *) imIn->image32[y],
                    imIn->xsize - 1, radius, edgeA, edgeB, ww, fw);
            }
        }
    }

    ImagingSectionLeave(&cookie);

    return imOut;
}


/* General implementation when radius > 0 and not too big */
void
ImagingInnerVertBoxBlur(Imaging imOut, Imaging imIn, int lasty, int radius,
                        int edgeA, int edgeB, UINT16 ww, UINT16 fw) 
{
    #define LINE 1024

    int x, xx, y;
    int line = LINE;
    INT16 acc[LINE];
    __m128i weights = _mm_set1_epi32((fw << 16) | ww);
    __m256i weights256 = _mm256_set1_epi32((fw << 16) | ww);
    
    UINT8 *lineOut, *lineAdd, *lineLeft, *lineRight;
    UINT8 *lineOutNext, *lineAddNext, *lineLeftNext, *lineRightNext;
    

    #define INNER_LOOP(line)  \
        x = 0; \
        for (; x < line - 31; x += 32) { \
            __m256i add = _mm256_loadu_si256((__m256i *)&lineAdd[x]); \
            __m256i add0 = _mm256_unpacklo_epi8(add, _mm256_setzero_si256()); \
            __m256i add1 = _mm256_unpackhi_epi8(add, _mm256_setzero_si256()); \
            __m256i left = _mm256_loadu_si256((__m256i *)&lineLeft[x]); \
            __m256i left0 = _mm256_unpacklo_epi8(left, _mm256_setzero_si256()); \
            __m256i left1 = _mm256_unpackhi_epi8(left, _mm256_setzero_si256()); \
            __m256i right = _mm256_loadu_si256((__m256i *)&lineRight[x]); \
            __m256i edge0 = _mm256_add_epi16(left0, \
                _mm256_unpacklo_epi8(right, _mm256_setzero_si256())); \
            __m256i edge1 = _mm256_add_epi16(left1, \
                _mm256_unpackhi_epi8(right, _mm256_setzero_si256())); \
            __m256i acc0 = _mm256_loadu_si256((__m256i *)&acc[x]); \
            __m256i acc1 = _mm256_loadu_si256((__m256i *)&acc[x+16]); \
            __m256i bulk0, bulk1, bulk2, bulk3; \
            acc0 = _mm256_add_epi16(_mm256_sub_epi16(acc0, left0), add0); \
            acc1 = _mm256_add_epi16(_mm256_sub_epi16(acc1, left1), add1); \
            _mm256_storeu_si256((__m256i *)&acc[x], acc0); \
            _mm256_storeu_si256((__m256i *)&acc[x+16], acc1); \
            bulk0 = _mm256_madd_epi16(weights256, _mm256_unpacklo_epi16(acc0, edge0)); \
            bulk1 = _mm256_madd_epi16(weights256, _mm256_unpackhi_epi16(acc0, edge0)); \
            bulk2 = _mm256_madd_epi16(weights256, _mm256_unpacklo_epi16(acc1, edge1)); \
            bulk3 = _mm256_madd_epi16(weights256, _mm256_unpackhi_epi16(acc1, edge1)); \
            bulk0 = _mm256_packs_epi32(_mm256_srli_epi32(bulk0, 18), \
                                       _mm256_srli_epi32(bulk1, 18)); \
            bulk2 = _mm256_packs_epi32(_mm256_srli_epi32(bulk2, 18), \
                                       _mm256_srli_epi32(bulk3, 18)); \
            _mm256_storeu_si256((__m256i *)&lineOut[x], \
                             _mm256_packus_epi16(bulk0, bulk2)); \
            _mm_prefetch(&lineOutNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineAddNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineLeftNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineRightNext[x], _MM_HINT_T0); \
        } \
        for (; x < line - 15; x += 16) { \
            __m128i add = _mm_loadu_si128((__m128i *)&lineAdd[x]); \
            __m128i add0 = _mm_unpacklo_epi8(add, _mm_setzero_si128()); \
            __m128i add1 = _mm_unpackhi_epi8(add, _mm_setzero_si128()); \
            __m128i left = _mm_loadu_si128((__m128i *)&lineLeft[x]); \
            __m128i left0 = _mm_unpacklo_epi8(left, _mm_setzero_si128()); \
            __m128i left1 = _mm_unpackhi_epi8(left, _mm_setzero_si128()); \
            __m128i right = _mm_loadu_si128((__m128i *)&lineRight[x]); \
            __m128i edge0 = _mm_add_epi16(left0, \
                _mm_unpacklo_epi8(right, _mm_setzero_si128())); \
            __m128i edge1 = _mm_add_epi16(left1, \
                _mm_unpackhi_epi8(right, _mm_setzero_si128())); \
            __m128i acc0 = _mm_loadu_si128((__m128i *)&acc[x]); \
            __m128i acc1 = _mm_loadu_si128((__m128i *)&acc[x+8]); \
            __m128i bulk0, bulk1, bulk2, bulk3; \
            acc0 = _mm_add_epi16(_mm_sub_epi16(acc0, left0), add0); \
            acc1 = _mm_add_epi16(_mm_sub_epi16(acc1, left1), add1); \
            _mm_storeu_si128((__m128i *)&acc[x], acc0); \
            _mm_storeu_si128((__m128i *)&acc[x+8], acc1); \
            bulk0 = _mm_madd_epi16(weights, _mm_unpacklo_epi16(acc0, edge0)); \
            bulk1 = _mm_madd_epi16(weights, _mm_unpackhi_epi16(acc0, edge0)); \
            bulk2 = _mm_madd_epi16(weights, _mm_unpacklo_epi16(acc1, edge1)); \
            bulk3 = _mm_madd_epi16(weights, _mm_unpackhi_epi16(acc1, edge1)); \
            bulk0 = _mm_packs_epi32(_mm_srli_epi32(bulk0, 18), \
                                    _mm_srli_epi32(bulk1, 18)); \
            bulk2 = _mm_packs_epi32(_mm_srli_epi32(bulk2, 18), \
                                    _mm_srli_epi32(bulk3, 18)); \
            _mm_storeu_si128((__m128i *)&lineOut[x], \
                             _mm_packus_epi16(bulk0, bulk2)); \
            _mm_prefetch(&lineOutNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineAddNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineLeftNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineRightNext[x], _MM_HINT_T0); \
        }

    for (xx = 0; xx < imIn->linesize; xx += LINE) {
        if (xx + LINE > imIn->linesize) {
            line = imIn->linesize - xx;
        }
        lineLeft = (UINT8 *)imIn->image[0] + xx;
        lineRight = (UINT8 *)imIn->image[lasty] + xx;
        for (x = 0; x < line; x++) {
            acc[x] = (lineLeft[x] * (radius + 1) +
                      lineRight[x] * (radius - edgeA + 1));
        }
        for (y = 0; y < edgeA - 1; y++) {
            lineAdd = (UINT8 *)imIn->image[y] + xx;
            for (x = 0; x < line; x++) {
                acc[x] += lineAdd[x];
            }
        }

        if (edgeA <= edgeB) {
            y = 0;
            lineOutNext = (UINT8 *)imOut->image[y] + xx;
            lineAddNext = (UINT8 *)imIn->image[y + radius] + xx;
            lineLeftNext = (UINT8 *)imIn->image[0] + xx;
            lineRightNext = (UINT8 *)imIn->image[y + radius + 1] + xx;
            for (; y < edgeA; y++) {
                lineOut = lineOutNext;
                lineAdd = lineAddNext;
                lineLeft = lineLeftNext;
                lineRight = lineRightNext;
                lineOutNext = (UINT8 *)imOut->image[y + 1] + xx;
                lineAddNext = (UINT8 *)imIn->image[y + 1 + radius] + xx;
                lineLeftNext = (UINT8 *)imIn->image[0] + xx;
                lineRightNext = (UINT8 *)imIn->image[y + 1 + radius + 1] + xx;
                INNER_LOOP(line);
            }
            lineOutNext = (UINT8 *)imOut->image[y] + xx;
            lineAddNext = (UINT8 *)imIn->image[y + radius] + xx;
            lineLeftNext = (UINT8 *)imIn->image[y - radius - 1] + xx;
            lineRightNext = (UINT8 *)imIn->image[y + radius + 1] + xx;
            for (; y < edgeB; y++) {
                lineOut = lineOutNext;
                lineAdd = lineAddNext;
                lineLeft = lineLeftNext;
                lineRight = lineRightNext;
                lineOutNext = (UINT8 *)imOut->image[y + 1] + xx;
                lineAddNext = (UINT8 *)imIn->image[y + 1 + radius] + xx;
                lineLeftNext = (UINT8 *)imIn->image[y + 1 - radius - 1] + xx;
                lineRightNext = (UINT8 *)imIn->image[y + 1 + radius + 1] + xx;
                INNER_LOOP(line);
            }
            for (; y <= lasty; y++) {
                lineOutNext = lineOut = (UINT8 *)imOut->image[y] + xx;
                lineLeftNext = lineLeft = (UINT8 *)imIn->image[y - radius - 1] + xx;
                lineAddNext = lineRightNext = lineAdd = lineRight = (UINT8 *)imIn->image[lasty] + xx;
                INNER_LOOP(line);
            }
        } else {
            y = 0;
            lineOutNext = (UINT8 *)imOut->image[y] + xx;
            lineAddNext = (UINT8 *)imIn->image[y + radius] + xx;
            lineLeftNext = (UINT8 *)imIn->image[0] + xx;
            lineRightNext = (UINT8 *)imIn->image[y + radius + 1] + xx;
            for (; y < edgeB; y++) {
                lineOut = lineOutNext;
                lineAdd = lineAddNext;
                lineLeft = lineLeftNext;
                lineRight = lineRightNext;
                lineOutNext = (UINT8 *)imOut->image[y + 1] + xx;
                lineAddNext = (UINT8 *)imIn->image[y + 1 + radius] + xx;
                lineLeftNext = (UINT8 *)imIn->image[0] + xx;
                lineRightNext = (UINT8 *)imIn->image[y + 1 + radius + 1] + xx;
                INNER_LOOP(line);
            }
            lineOutNext = (UINT8 *)imOut->image[y] + xx;
            lineLeftNext = (UINT8 *)imIn->image[0] + xx;
            lineAddNext = lineRight = (UINT8 *)imIn->image[lasty] + xx;
            for (; y < edgeA; y++) {
                lineOut = lineOutNext;
                lineAdd = lineAddNext;
                lineLeft = lineLeftNext;
                lineRight = lineRightNext;
                lineOutNext = (UINT8 *)imOut->image[y + 1] + xx;
                lineLeftNext = (UINT8 *)imIn->image[0] + xx;
                lineAddNext = lineRightNext = (UINT8 *)imIn->image[lasty] + xx;
                INNER_LOOP(line);
            }
            for (; y <= lasty; y++) {
                lineOutNext = lineOut = (UINT8 *)imOut->image[y] + xx;
                lineLeftNext = lineLeft = (UINT8 *)imIn->image[y - radius - 1] + xx;
                lineAddNext = lineRightNext = lineAdd = lineRight = (UINT8 *)imIn->image[lasty] + xx;
                INNER_LOOP(line);
            }
        }
    }

    #undef INNER_LOOP
    #undef LINE
}


/* Implementation for large or integer radii */
void
ImagingInnerVertBoxBlurLarge(Imaging imOut, Imaging imIn, int lasty, int radius,
                             int edgeA, int edgeB, UINT16 ww) 
{
    #define LINE 1024

    int x, xx, y;
    int line = LINE;
    UINT32 acc[LINE];
    __m128i weights = _mm_set1_epi16(ww);
    __m256i weights256 = _mm256_set1_epi16(ww);
    
    UINT8 *lineOut, *lineAdd, *lineLeft;
    UINT8 *lineOutNext, *lineAddNext, *lineLeftNext;
    

    #define INNER_LOOP(line)  \
        x = 0; \
        for (; x < line - 31; x += 32) { \
            __m256i add = _mm256_loadu_si256((__m256i *)&lineAdd[x]); \
            __m256i add0 = _mm256_unpacklo_epi8(add, _mm256_setzero_si256()); \
            __m256i add1 = _mm256_unpackhi_epi8(add, _mm256_setzero_si256()); \
            __m256i left = _mm256_loadu_si256((__m256i *)&lineLeft[x]); \
            __m256i left0 = _mm256_unpacklo_epi8(left, _mm256_setzero_si256()); \
            __m256i left1 = _mm256_unpackhi_epi8(left, _mm256_setzero_si256()); \
            __m256i acc0 = _mm256_loadu_si256((__m256i *)&acc[x]); \
            __m256i acc1 = _mm256_loadu_si256((__m256i *)&acc[x+16]); \
            __m256i bulk0, bulk1; \
            acc0 = _mm256_add_epi16(_mm256_sub_epi16(acc0, left0), add0); \
            acc1 = _mm256_add_epi16(_mm256_sub_epi16(acc1, left1), add1); \
            _mm256_storeu_si256((__m256i *)&acc[x], acc0); \
            _mm256_storeu_si256((__m256i *)&acc[x+16], acc1); \
            bulk0 = _mm256_mulhi_epu16(weights256, acc0); \
            bulk1 = _mm256_mulhi_epu16(weights256, acc0); \
            bulk0 = _mm256_packus_epi16(_mm256_srli_epi32(bulk0, 2), \
                                     _mm256_srli_epi32(bulk1, 2)); \
            _mm256_storeu_si256((__m256i *)&lineOut[x], bulk0); \
            _mm_prefetch(&lineOutNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineAddNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineLeftNext[x], _MM_HINT_T0); \
        } \
        for (; x < line - 15; x += 16) { \
            __m128i add = _mm_loadu_si128((__m128i *)&lineAdd[x]); \
            __m128i add0 = _mm_unpacklo_epi8(add, _mm_setzero_si128()); \
            __m128i add1 = _mm_unpackhi_epi8(add, _mm_setzero_si128()); \
            __m128i left = _mm_loadu_si128((__m128i *)&lineLeft[x]); \
            __m128i left0 = _mm_unpacklo_epi8(left, _mm_setzero_si128()); \
            __m128i left1 = _mm_unpackhi_epi8(left, _mm_setzero_si128()); \
            __m128i acc0 = _mm_loadu_si128((__m128i *)&acc[x]); \
            __m128i acc1 = _mm_loadu_si128((__m128i *)&acc[x+8]); \
            __m128i bulk0, bulk1; \
            acc0 = _mm_add_epi16(_mm_sub_epi16(acc0, left0), add0); \
            acc1 = _mm_add_epi16(_mm_sub_epi16(acc1, left1), add1); \
            _mm_storeu_si128((__m128i *)&acc[x], acc0); \
            _mm_storeu_si128((__m128i *)&acc[x+8], acc1); \
            bulk0 = _mm_mulhi_epu16(weights, acc0); \
            bulk1 = _mm_mulhi_epu16(weights, acc0); \
            bulk0 = _mm_packus_epi16(_mm_srli_epi32(bulk0, 2), \
                                     _mm_srli_epi32(bulk1, 2)); \
            _mm_storeu_si128((__m128i *)&lineOut[x], bulk0); \
            _mm_prefetch(&lineOutNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineAddNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineLeftNext[x], _MM_HINT_T0); \
        }

    for (xx = 0; xx < imIn->linesize; xx += LINE) {
        if (xx + LINE > imIn->linesize) {
            line = imIn->linesize - xx;
        }
        lineLeft = (UINT8 *)imIn->image[0] + xx;
        lineAdd = (UINT8 *)imIn->image[lasty] + xx;
        for (x = 0; x < line; x++) {
            acc[x] = (lineLeft[x] * (radius + 1) +
                      lineAdd[x] * (radius - edgeA + 1));
        }
        for (y = 0; y < edgeA - 1; y++) {
            lineAdd = (UINT8 *)imIn->image[y] + xx;
            for (x = 0; x < line; x++) {
                acc[x] += lineAdd[x];
            }
        }

        if (edgeA <= edgeB) {
            y = 0;
            lineOutNext = (UINT8 *)imOut->image[y] + xx;
            lineAddNext = (UINT8 *)imIn->image[y + radius] + xx;
            lineLeftNext = (UINT8 *)imIn->image[0] + xx;
            for (; y < edgeA; y++) {
                lineOut = lineOutNext;
                lineAdd = lineAddNext;
                lineLeft = lineLeftNext;
                lineOutNext = (UINT8 *)imOut->image[y + 1] + xx;
                lineAddNext = (UINT8 *)imIn->image[y + 1 + radius] + xx;
                lineLeftNext = (UINT8 *)imIn->image[0] + xx;
                INNER_LOOP(line);
            }
            lineOutNext = (UINT8 *)imOut->image[y] + xx;
            lineAddNext = (UINT8 *)imIn->image[y + radius] + xx;
            lineLeftNext = (UINT8 *)imIn->image[y - radius - 1] + xx;
            for (; y < edgeB; y++) {
                lineOut = lineOutNext;
                lineAdd = lineAddNext;
                lineLeft = lineLeftNext;
                lineOutNext = (UINT8 *)imOut->image[y + 1] + xx;
                lineAddNext = (UINT8 *)imIn->image[y + 1 + radius] + xx;
                lineLeftNext = (UINT8 *)imIn->image[y + 1 - radius - 1] + xx;
                INNER_LOOP(line);
            }
            for (; y <= lasty; y++) {
                lineOutNext = lineOut = (UINT8 *)imOut->image[y] + xx;
                lineLeftNext = lineLeft = (UINT8 *)imIn->image[y - radius - 1] + xx;
                lineAddNext = lineAdd = (UINT8 *)imIn->image[lasty] + xx;
                INNER_LOOP(line);
            }
        } else {
            y = 0;
            lineOutNext = (UINT8 *)imOut->image[y] + xx;
            lineAddNext = (UINT8 *)imIn->image[y + radius] + xx;
            lineLeftNext = (UINT8 *)imIn->image[0] + xx;
            for (; y < edgeB; y++) {
                lineOut = lineOutNext;
                lineAdd = lineAddNext;
                lineLeft = lineLeftNext;
                lineOutNext = (UINT8 *)imOut->image[y + 1] + xx;
                lineAddNext = (UINT8 *)imIn->image[y + 1 + radius] + xx;
                lineLeftNext = (UINT8 *)imIn->image[0] + xx;
                INNER_LOOP(line);
            }
            lineOutNext = (UINT8 *)imOut->image[y] + xx;
            lineLeftNext = (UINT8 *)imIn->image[0] + xx;
            lineAddNext = (UINT8 *)imIn->image[lasty] + xx;
            for (; y < edgeA; y++) {
                lineOut = lineOutNext;
                lineAdd = lineAddNext;
                lineLeft = lineLeftNext;
                lineOutNext = (UINT8 *)imOut->image[y + 1] + xx;
                lineLeftNext = (UINT8 *)imIn->image[0] + xx;
                lineAddNext = (UINT8 *)imIn->image[lasty] + xx;
                INNER_LOOP(line);
            }
            for (; y <= lasty; y++) {
                lineOutNext = lineOut = (UINT8 *)imOut->image[y] + xx;
                lineLeftNext = lineLeft = (UINT8 *)imIn->image[y - radius - 1] + xx;
                lineAddNext = lineAdd = (UINT8 *)imIn->image[lasty] + xx;
                INNER_LOOP(line);
            }
        }
    }

    #undef INNER_LOOP
    #undef LINE
}


/* Optimized implementation when radius = 0 */
void
ImagingInnerVertBoxBlurZero(Imaging imOut, Imaging imIn, int lasty,
                            int edgeA, int edgeB, UINT16 ww, UINT16 fw)
{
    #define LINE 1024
    
    int x, xx, y;
    int line = LINE;
    __m128i weights = _mm_set1_epi32((fw << 16) | ww);
    __m256i weights256 = _mm256_set1_epi32((fw << 16) | ww);
    
    UINT8 *lineOut, *lineAdd, *lineLeft, *lineRight;
    UINT8 *lineOutNext, *lineAddNext, *lineLeftNext, *lineRightNext;
    

    #define INNER_LOOP(line)  \
        x = 0; \
        for (; x < line - 31; x += 32) { \
            __m256i add = _mm256_loadu_si256((__m256i *)&lineAdd[x]); \
            __m256i add0 = _mm256_unpacklo_epi8(add, _mm256_setzero_si256()); \
            __m256i add1 = _mm256_unpackhi_epi8(add, _mm256_setzero_si256()); \
            __m256i left = _mm256_loadu_si256((__m256i *)&lineLeft[x]); \
            __m256i right = _mm256_loadu_si256((__m256i *)&lineRight[x]); \
            __m256i edge0 = _mm256_add_epi16( \
                _mm256_unpacklo_epi8(left, _mm256_setzero_si256()), \
                _mm256_unpacklo_epi8(right, _mm256_setzero_si256())); \
            __m256i edge1 = _mm256_add_epi16( \
                _mm256_unpackhi_epi8(left, _mm256_setzero_si256()), \
                _mm256_unpackhi_epi8(right, _mm256_setzero_si256())); \
            __m256i bulk0, bulk1, bulk2, bulk3; \
            bulk0 = _mm256_madd_epi16(weights256, _mm256_unpacklo_epi16(add0, edge0)); \
            bulk1 = _mm256_madd_epi16(weights256, _mm256_unpackhi_epi16(add0, edge0)); \
            bulk2 = _mm256_madd_epi16(weights256, _mm256_unpacklo_epi16(add1, edge1)); \
            bulk3 = _mm256_madd_epi16(weights256, _mm256_unpackhi_epi16(add1, edge1)); \
            bulk0 = _mm256_packs_epi32(_mm256_srli_epi32(bulk0, 18), \
                                       _mm256_srli_epi32(bulk1, 18)); \
            bulk2 = _mm256_packs_epi32(_mm256_srli_epi32(bulk2, 18), \
                                       _mm256_srli_epi32(bulk3, 18)); \
            _mm256_storeu_si256((__m256i *)&lineOut[x], \
                                _mm256_packus_epi16(bulk0, bulk2)); \
            _mm_prefetch(&lineOutNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineAddNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineLeftNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineRightNext[x], _MM_HINT_T0); \
        } \
        for (; x < line - 15; x += 16) { \
            __m128i add = _mm_loadu_si128((__m128i *)&lineAdd[x]); \
            __m128i add0 = _mm_unpacklo_epi8(add, _mm_setzero_si128()); \
            __m128i add1 = _mm_unpackhi_epi8(add, _mm_setzero_si128()); \
            __m128i left = _mm_loadu_si128((__m128i *)&lineLeft[x]); \
            __m128i right = _mm_loadu_si128((__m128i *)&lineRight[x]); \
            __m128i edge0 = _mm_add_epi16( \
                _mm_unpacklo_epi8(left, _mm_setzero_si128()), \
                _mm_unpacklo_epi8(right, _mm_setzero_si128())); \
            __m128i edge1 = _mm_add_epi16( \
                _mm_unpackhi_epi8(left, _mm_setzero_si128()), \
                _mm_unpackhi_epi8(right, _mm_setzero_si128())); \
            __m128i bulk0, bulk1, bulk2, bulk3; \
            bulk0 = _mm_madd_epi16(weights, _mm_unpacklo_epi16(add0, edge0)); \
            bulk1 = _mm_madd_epi16(weights, _mm_unpackhi_epi16(add0, edge0)); \
            bulk2 = _mm_madd_epi16(weights, _mm_unpacklo_epi16(add1, edge1)); \
            bulk3 = _mm_madd_epi16(weights, _mm_unpackhi_epi16(add1, edge1)); \
            bulk0 = _mm_packs_epi32(_mm_srli_epi32(bulk0, 18), \
                                    _mm_srli_epi32(bulk1, 18)); \
            bulk2 = _mm_packs_epi32(_mm_srli_epi32(bulk2, 18), \
                                    _mm_srli_epi32(bulk3, 18)); \
            _mm_storeu_si128((__m128i *)&lineOut[x], \
                             _mm_packus_epi16(bulk0, bulk2)); \
            _mm_prefetch(&lineOutNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineAddNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineLeftNext[x], _MM_HINT_T0); \
            _mm_prefetch(&lineRightNext[x], _MM_HINT_T0); \
        }

    for (xx = 0; xx < imIn->linesize - 15; xx += LINE) {
        if (xx + LINE > imIn->linesize) {
            line = imIn->linesize - xx;
        }

        if (edgeA <= edgeB) {
            for (y = 0; y < edgeA; y++) {
                lineOutNext = lineOut = (UINT8 *)imOut->image[y] + xx;
                lineAddNext = lineAdd = (UINT8 *)imIn->image[y] + xx;
                lineLeftNext = lineLeft = (UINT8 *)imIn->image[0] + xx;
                lineRightNext = lineRight = (UINT8 *)imIn->image[y + 1] + xx;
                INNER_LOOP(line);
            }
            lineOutNext = (UINT8 *)imOut->image[y] + xx;
            lineAddNext = (UINT8 *)imIn->image[y] + xx;
            lineLeftNext = (UINT8 *)imIn->image[y - 1] + xx;
            lineRightNext = (UINT8 *)imIn->image[y + 1] + xx;
            for (y = edgeA; y < edgeB; y++) {
                lineOut = lineOutNext;
                lineAdd = lineAddNext;
                lineLeft = lineLeftNext;
                lineRight = lineRightNext;
                lineOutNext = (UINT8 *)imOut->image[y + 1] + xx;
                lineAddNext = (UINT8 *)imIn->image[y + 1] + xx;
                lineLeftNext = (UINT8 *)imIn->image[y + 1 - 1] + xx;
                lineRightNext = (UINT8 *)imIn->image[y + 1 + 1] + xx;
                INNER_LOOP(line);
            }
            lineOutNext = (UINT8 *)imOut->image[0] + xx*2;
            lineAddNext = (UINT8 *)imIn->image[0] + xx*2;
            lineLeftNext = (UINT8 *)imIn->image[0] + xx*2;
            lineRightNext = (UINT8 *)imIn->image[1] + xx*2;
            for (y = edgeB; y <= lasty; y++) {
                lineOut = (UINT8 *)imOut->image[y] + xx;
                lineLeft = (UINT8 *)imIn->image[y - 1] + xx;
                lineAdd = lineRight = (UINT8 *)imIn->image[lasty] + xx;
                INNER_LOOP(line);
            }
        } else {
            for (y = edgeB; y < edgeA; y++) {
                lineOutNext = lineOut = (UINT8 *)imOut->image[y] + xx;
                lineLeftNext = lineLeft = (UINT8 *)imIn->image[0] + xx;
                lineAddNext = lineRightNext = lineAdd = lineRight = (UINT8 *)imIn->image[lasty] + xx;
                INNER_LOOP(line);
            }
        }
    }

    #undef INNER_LOOP
    #undef LINE
}


Imaging
ImagingVerticalBoxBlur(Imaging imOut, Imaging imIn, float fRadius)
{
    ImagingSectionCookie cookie;

    UINT32 ww, fw;
    int edgeA, edgeB;
    int radius = compute_params(fRadius, imIn->ysize, &edgeA, &edgeB, &ww, &fw);
    int lasty = imIn->ysize - 1;

    // printf(">>> %d, from %d to %d. %d %d\n", radius, edgeA, edgeB, ww, fw);

    ImagingSectionEnter(&cookie);

    if (radius == 0) {
        ImagingInnerVertBoxBlurZero(imOut, imIn, lasty,
                                    edgeA, edgeB, ww, fw);
    } else if (fw == 0) {
        ImagingInnerVertBoxBlurLarge(imOut, imIn, lasty, radius,
                                     edgeA, edgeB, ww);
    } else {
        ImagingInnerVertBoxBlur(imOut, imIn, lasty, radius,
                                edgeA, edgeB, ww, fw);
    }

    ImagingSectionLeave(&cookie);

    return imOut;
}


Imaging
ImagingBoxBlur(Imaging imOut, Imaging imIn, float radius, int n)
{
    int i;
    /* Real allocated buffer image */
    Imaging imBuffer;
    /* Pointers to current working images */
    Imaging imFrom, imTo;
    /* Temp pointer for swapping current working images */
    Imaging imTemp;

    if (n < 1) {
        return ImagingError_ValueError(
            "number of passes must be greater than zero");
    }
    if (radius >= (1<<ACC_MAX_PRECISION)) {
        return ImagingError_ValueError("Radius is too large");
    }
    if (radius < 0 ) {
        return ImagingError_ValueError("Radius can't be negative");
    }

    if (strcmp(imIn->mode, imOut->mode) ||
        imIn->type  != imOut->type  ||
        imIn->bands != imOut->bands ||
        imIn->xsize != imOut->xsize ||
        imIn->ysize != imOut->ysize)
        return ImagingError_Mismatch();

    if (imIn->type != IMAGING_TYPE_UINT8)
        return ImagingError_ModeError();

    if (!(strcmp(imIn->mode, "RGB") == 0 ||
          strcmp(imIn->mode, "RGBA") == 0 ||
          strcmp(imIn->mode, "RGBa") == 0 ||
          strcmp(imIn->mode, "RGBX") == 0 ||
          strcmp(imIn->mode, "CMYK") == 0 ||
          strcmp(imIn->mode, "L") == 0 ||
          strcmp(imIn->mode, "LA") == 0 ||
          strcmp(imIn->mode, "La") == 0))
        return ImagingError_ModeError();

    imBuffer = ImagingNewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if ( ! imBuffer)
        return NULL;

    /* We always do 2n passes. Odd to imBuffer, even to imOut */
    imTo = imBuffer;
    imFrom = imOut;

    /* First pass, use imIn instead of imFrom. */
    ImagingVerticalBoxBlur(imTo, imIn, radius);
    for (i = 1; i < n; i ++) {
        /* Swap current working images */
        imTemp = imTo; imTo = imFrom; imFrom = imTemp;
        ImagingVerticalBoxBlur(imTo, imFrom, radius);
    }

    /* Reuse imOut as a source and destination there. */
    for (i = 0; i < n; i ++) {
        /* Swap current working images */
        imTemp = imTo; imTo = imFrom; imFrom = imTemp;
        ImagingVerticalBoxBlur(imTo, imFrom, radius);
    }

    ImagingDelete(imBuffer);

    return imOut;
}


Imaging ImagingGaussianBlur(Imaging imOut, Imaging imIn, float radius,
    int passes)
{
    float sigma2, L, l, a;

    sigma2 = radius * radius / passes;
    // from http://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf
    // [7] Box length.
    L = sqrt(12.0 * sigma2 + 1.0);
    // [11] Integer part of box radius.
    l = floor((L - 1.0) / 2.0);
    // [14], [Fig. 2] Fractional part of box radius.
    a = (2 * l + 1) * (l * (l + 1) - 3 * sigma2);
    a /= 6 * (sigma2 - (l + 1) * (l + 1));

    return ImagingBoxBlur(imOut, imIn, l + a, passes);
}

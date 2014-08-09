/**
 * @date Tue Nov 22 11:24:44 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implementation of row-major/column-major reordering
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_IO_BASE_MODULE
#include <bob.io.base/api.h>
#include <bob.blitz/capi.h>
#include <cstring> //for memcpy

/**
 * Returns, on the first argument, the linear indexes by calculating the
 * linear positions relative to both row-major and column-major order
 * matrixes given a certain index accessing a position in the matrix and the
 * matrix shape
 *
 * @param row The resulting row-major linear index.
 *            (row,col) is a 2-tuple with the results: row-major and
 *            column-major linear indexes
 * @param col The resulting column-major linear index. (see above)
 * @param i   Index of the column.
 *            (i,j) a 2-tuple with the indexes as would be accessed
 *            [col][row]; this is the same as accessing the matrix like
 *            on directions [y][x]
 * @param j   Index of the row. (see above)
 * @param shape a 2-tuple with the matrix shape like [col][row]; this is the
 *        same as thinking about the extends of the matrix like on directions
 *        [y][x]
 *
 * Detailed arithmetics with graphics and explanations can be found here:
 * http://webster.cs.ucr.edu/AoA/Windows/HTML/Arraysa2.html
 */
static void rc2d(size_t& row, size_t& col, const size_t i, const size_t j,
    const size_t* shape) {

  row = (i * shape[1]) + j;
  col = (j * shape[0]) + i;

}

/**
 * Same as above, but for a 3D array organized as [depth][column][row]
 */
static void rc3d(size_t& row, size_t& col, const size_t i, const size_t j,
    const size_t k, const size_t* shape) {

  row = ( (i * shape[1]) + j ) * shape[2] + k;
  col = ( (k * shape[1]) + j ) * shape[0] + i;

}

/**
 * Same as above, but for a 4D array organized as [time][depth][column][row]
 */
static void rc4d(size_t& row, size_t& col, const size_t i, const size_t j,
    const size_t k, const size_t l, const size_t* shape) {

  row = ( ( i * shape[1] + j ) * shape[2] + k ) * shape[3] + l;
  col = ( ( l * shape[2] + k ) * shape[1] + j ) * shape[0] + i;

}

int BobIoReorder_RowToCol(const void* src_, void* dst_,
    const BobIoTypeinfo* info) {

  size_t dsize = PyBlitzArray_TypenumSize(info->dtype);

  //cast to byte type so we can manipulate the pointers...
  const uint8_t* src = static_cast<const uint8_t*>(src_);
  uint8_t* dst = static_cast<uint8_t*>(dst_);

  switch(info->nd) {

    case 1:
      std::memcpy(dst, src, BobIoTypeinfo_BufferSize(info));
      break;

    case 2:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j) {
          size_t row_major, col_major;
          rc2d(row_major, col_major, i, j, info->shape);
          row_major *= dsize;
          col_major *= dsize;
          std::memcpy(&dst[col_major], &src[row_major], dsize);
        }
      break;

    case 3:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j)
          for (size_t k=0; k<info->shape[2]; ++k) {
            size_t row_major, col_major;
            rc3d(row_major, col_major, i, j, k, info->shape);
            row_major *= dsize;
            col_major *= dsize;
            std::memcpy(&dst[col_major], &src[row_major], dsize);
          }
      break;

    case 4:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j)
          for (size_t k=0; k<info->shape[2]; ++k)
            for (size_t l=0; l<info->shape[3]; ++l) {
              size_t row_major, col_major;
              rc4d(row_major, col_major, i, j, k, l, info->shape);
              row_major *= dsize;
              col_major *= dsize;
              std::memcpy(&dst[col_major], &src[row_major], dsize);
            }
      break;

    default:
      PyErr_Format(PyExc_RuntimeError, "can only flip arrays with up to %u dimensions - you passed one with %zu dimensions", BOB_BLITZ_MAXDIMS, info->nd);
      return 0;
  }

  return 1;
}

int BobIoReorder_ColToRow(const void* src_, void* dst_,
    const BobIoTypeinfo* info) {

  size_t dsize = PyBlitzArray_TypenumSize(info->dtype);

  //cast to byte type so we can manipulate the pointers...
  const uint8_t* src = static_cast<const uint8_t*>(src_);
  uint8_t* dst = static_cast<uint8_t*>(dst_);

  switch(info->nd) {

    case 1:
      std::memcpy(dst, src, BobIoTypeinfo_BufferSize(info));
      break;

    case 2:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j) {
          size_t row_major, col_major;
          rc2d(row_major, col_major, i, j, info->shape);
          row_major *= dsize;
          col_major *= dsize;
          std::memcpy(&dst[row_major], &src[col_major], dsize);
        }
      break;

    case 3:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j)
          for (size_t k=0; k<info->shape[2]; ++k) {
            size_t row_major, col_major;
            rc3d(row_major, col_major, i, j, k, info->shape);
            row_major *= dsize;
            col_major *= dsize;
            std::memcpy(&dst[row_major], &src[col_major], dsize);
          }
      break;

    case 4:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j)
          for (size_t k=0; k<info->shape[2]; ++k)
            for (size_t l=0; l<info->shape[3]; ++l) {
              size_t row_major, col_major;
              rc4d(row_major, col_major, i, j, k, l, info->shape);
              row_major *= dsize;
              col_major *= dsize;
              std::memcpy(&dst[row_major], &src[col_major], dsize);
            }
      break;

    default:
      PyErr_Format(PyExc_RuntimeError, "can only flip arrays with up to %u dimensions - you passed one with %zu dimensions", BOB_BLITZ_MAXDIMS, info->nd);
      return 0;
  }

  return 1;
}

int BobIoReorder_RowToColComplex(const void* src_, void* dst_re_,
    void* dst_im_, const BobIoTypeinfo* info) {

  size_t dsize = PyBlitzArray_TypenumSize(info->dtype);
  size_t dsize2 = dsize/2; ///< size of each complex component (real, imaginary)

  //cast to byte type so we can manipulate the pointers...
  const uint8_t* src = static_cast<const uint8_t*>(src_);
  uint8_t* dst_re = static_cast<uint8_t*>(dst_re_);
  uint8_t* dst_im = static_cast<uint8_t*>(dst_im_);

  switch(info->nd) {

    case 1:
      for (size_t i=0; i<info->shape[0]; ++i) {
        std::memcpy(&dst_re[dsize2*i], &src[dsize*i]       , dsize2);
        std::memcpy(&dst_im[dsize2*i], &src[dsize*i]+dsize2, dsize2);
      }
      break;

    case 2:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j) {
          size_t row_major, col_major;
          rc2d(row_major, col_major, i, j, info->shape);
          row_major *= dsize;
          col_major *= dsize2;
          std::memcpy(&dst_re[col_major], &src[row_major]       , dsize2);
          std::memcpy(&dst_im[col_major], &src[row_major]+dsize2, dsize2);
        }
      break;

    case 3:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j)
          for (size_t k=0; k<info->shape[2]; ++k) {
            size_t row_major, col_major;
            rc3d(row_major, col_major, i, j, k, info->shape);
            row_major *= dsize;
            col_major *= dsize2;
            std::memcpy(&dst_re[col_major], &src[row_major]       , dsize2);
            std::memcpy(&dst_im[col_major], &src[row_major]+dsize2, dsize2);
          }
      break;

    case 4:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j)
          for (size_t k=0; k<info->shape[2]; ++k)
            for (size_t l=0; l<info->shape[3]; ++l) {
              size_t row_major, col_major;
              rc4d(row_major, col_major, i, j, k, l, info->shape);
              row_major *= dsize;
              col_major *= dsize2;
              std::memcpy(&dst_re[col_major], &src[row_major]       , dsize2);
              std::memcpy(&dst_im[col_major], &src[row_major]+dsize2, dsize2);
            }
      break;

    default:
      PyErr_Format(PyExc_RuntimeError, "can only flip arrays with up to %u dimensions - you passed one with %zu dimensions", BOB_BLITZ_MAXDIMS, info->nd);
      return 0;
  }

  return 1;
}

int BobIoReorder_ColToRowComplex(const void* src_re_,
    const void* src_im_, void* dst_, const BobIoTypeinfo* info) {

  size_t dsize = PyBlitzArray_TypenumSize(info->dtype);
  size_t dsize2 = dsize/2; ///< size of each complex component (real, imaginary)

  //cast to byte type so we can manipulate the pointers...
  const uint8_t* src_re = static_cast<const uint8_t*>(src_re_);
  const uint8_t* src_im = static_cast<const uint8_t*>(src_im_);
  uint8_t* dst = static_cast<uint8_t*>(dst_);

  switch(info->nd) {

    case 1:
      for (size_t i=0; i<info->shape[0]; ++i) {
        std::memcpy(&dst[dsize*i]       , &src_re[dsize2*i], dsize2);
        std::memcpy(&dst[dsize*i]+dsize2, &src_im[dsize2*i], dsize2);
      }
      break;

    case 2:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j) {
          size_t row_major, col_major;
          rc2d(row_major, col_major, i, j, info->shape);
          row_major *= dsize;
          col_major *= dsize2;
          std::memcpy(&dst[row_major],        &src_re[col_major], dsize2);
          std::memcpy(&dst[row_major]+dsize2, &src_im[col_major], dsize2);
        }
      break;

    case 3:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j)
          for (size_t k=0; k<info->shape[2]; ++k) {
            size_t row_major, col_major;
            rc3d(row_major, col_major, i, j, k, info->shape);
            row_major *= dsize;
            col_major *= dsize2;
            std::memcpy(&dst[row_major]       , &src_re[col_major], dsize2);
            std::memcpy(&dst[row_major]+dsize2, &src_im[col_major], dsize2);
          }
      break;

    case 4:
      for (size_t i=0; i<info->shape[0]; ++i)
        for (size_t j=0; j<info->shape[1]; ++j)
          for (size_t k=0; k<info->shape[2]; ++k)
            for (size_t l=0; l<info->shape[3]; ++l) {
              size_t row_major, col_major;
              rc4d(row_major, col_major, i, j, k, l, info->shape);
              row_major *= dsize;
              col_major *= dsize2;
              std::memcpy(&dst[row_major]       , &src_re[col_major], dsize2);
              std::memcpy(&dst[row_major]+dsize2, &src_im[col_major], dsize2);
            }
      break;

    default:
      PyErr_Format(PyExc_RuntimeError, "can only flip arrays with up to %u dimensions - you passed one with %zu dimensions", BOB_BLITZ_MAXDIMS, info->nd);
      return 0;
  }

  return 1;
}

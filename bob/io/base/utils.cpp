/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  3 Oct 08:36:48 2012
 *
 * @brief Implementation of some compile-time I/O utitlites
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_IO_BASE_MODULE
#include <bob.io.base/api.h>
#include <bob.io.base/File.h>
#include "cpp/CodecRegistry.h"

boost::shared_ptr<bob::io::base::File> BobIoFile_Open
  (const char* filename, char mode) {

  boost::shared_ptr<bob::io::base::CodecRegistry> instance = bob::io::base::CodecRegistry::instance();
  return instance->findByFilenameExtension(filename)(filename, mode);

}

boost::shared_ptr<bob::io::base::File> BobIoFile_OpenWithExtension (const char* filename,
    char mode, const char* pretend_extension) {

  boost::shared_ptr<bob::io::base::CodecRegistry> instance = bob::io::base::CodecRegistry::instance();
  return instance->findByExtension(pretend_extension)(filename, mode);

}

void BobIoFile_Peek (const char* filename, BobIoTypeinfo* info) {
  BobIoTypeinfo_Copy(info, &BobIoFile_Open(filename, 'r')->type());
}

void BobIoFile_PeekAll (const char* filename, BobIoTypeinfo* info) {
  BobIoTypeinfo_Copy(info, &BobIoFile_Open(filename, 'r')->type_all());
}

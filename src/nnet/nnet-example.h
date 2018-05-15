// nnet0/nnet-example.h

// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef NNET_NNET_EXAMPLE_H_
#define NNET_NNET_EXAMPLE_H_

namespace kaldi {

using namespace std;

class Mutex {
 public:
  Mutex() {}

  inline void Lock() { mu_.lock(); }

  inline void Unlock() { mu_.unlock(); }

 private:
  std::mutex mu_;

  Mutex(const Mutex &) = delete;
  Mutex &operator=(const Mutex &) = delete;
};

class MutexLock {
 public:
  explicit MutexLock(Mutex *mu) : mu_(mu) { mu_->Lock(); }

  ~MutexLock() { mu_->Unlock(); }

 private:
  Mutex *mu_;

  MutexLock(const MutexLock &) = delete;
  MutexLock &operator=(const MutexLock &) = delete;
};

namespace nnet0 {

struct NnetExample{

  SequentialBaseFloatMatrixReader *feature_reader;

  std::string utt;
  Matrix<BaseFloat> input_frames;

  NnetExample(SequentialBaseFloatMatrixReader *feature_reader):
    feature_reader(feature_reader){}

  virtual ~NnetExample(){}

  virtual bool PrepareData(std::vector<NnetExample*> &examples) = 0;


};

struct FeatureExample: NnetExample
{
  FeatureExample(SequentialBaseFloatMatrixReader *feature_reader)
  :NnetExample(feature_reader){}

  bool PrepareData(std::vector<NnetExample*> &examples)
  {
    examples.resize(1);
    utt = feature_reader->Key();
    input_frames = feature_reader->Value();
    examples[0] = this;
    return true;
  }

};

/** This struct stores neural net training examples to be used in
    multi-threaded training.  */
class ExamplesRepository {
 public:
  /// The following function is called by the code that reads in the examples.
  void AcceptExample(NnetExample *example) {
  empty_semaphore_.Wait();
  examples_mutex_.Lock();
  examples_.push_back(example);
  examples_mutex_.Unlock();
  full_semaphore_.Signal();
}

  /// The following function is called by the code that reads in the examples,
  /// when we're done reading examples; it signals this way to this class
  /// that the stream is now empty
  void ExamplesDone() {
  for (int32 i = 0; i < buffer_size_; i++)
    empty_semaphore_.Wait();
  examples_mutex_.Lock();
  KALDI_ASSERT(examples_.empty());
  examples_mutex_.Unlock();
  done_ = true;
  full_semaphore_.Signal();
}

  /// This function is called by the code that does the training.  If there is
  /// an example available it will provide it, or it will sleep till one is
  /// available.  It returns NULL when there are no examples left and
  /// ExamplesDone() has been called.
  NnetExample *ProvideExample() {
  full_semaphore_.Wait();
  if (done_) {
    KALDI_ASSERT(examples_.empty());
    full_semaphore_.Signal(); // Increment the semaphore so
    // the call by the next thread will not block.
    return NULL; // no examples to return-- all finished.
  } else {
    examples_mutex_.Lock();
    KALDI_ASSERT(!examples_.empty());
    NnetExample *ans = examples_.front();
    examples_.pop_front();
    examples_mutex_.Unlock();
    empty_semaphore_.Signal();
    return ans;
  }
}

  ExamplesRepository(int32 buffer_size = 128): buffer_size_(buffer_size),
                                      empty_semaphore_(buffer_size_),
                                      done_(false) { }
 private:
  int32 buffer_size_;
  Semaphore full_semaphore_;
  Semaphore empty_semaphore_;
  Mutex examples_mutex_; // mutex we lock to modify examples_.

  std::deque<NnetExample*> examples_;
  bool done_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(ExamplesRepository);
};

} // namespace nnet
} // namespace kaldi


#endif /* NNET_NNET_EXAMPLE_H_ */


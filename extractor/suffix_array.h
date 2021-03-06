#ifndef _SUFFIX_ARRAY_H_
#define _SUFFIX_ARRAY_H_

#include <memory>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using namespace std;

namespace extractor {

class DataArray;
class PhraseLocation;

class SuffixArray {
 public:
  // Creates a suffix array from a data array.
  SuffixArray(shared_ptr<DataArray> data_array);

  virtual ~SuffixArray();

  // Returns the size of the suffix array.
  virtual int GetSize() const;

  // Returns the data array on top of which the suffix array is constructed.
  virtual shared_ptr<DataArray> GetData() const;

  // Constructs the longest-common-prefix array using the algorithm of Kasai et
  // al. (2001).
  virtual vector<int> BuildLCPArray() const;

  // Returns the i-th suffix.
  virtual int GetSuffix(int rank) const;

  // Given the range in which a phrase is located and the next word, returns the
  // range corresponding to the phrase extended with the next word.
  virtual PhraseLocation Lookup(int low, int high, const string& word,
                                int offset) const;

  void WriteBinary(const fs::path& filepath) const;

 protected:
  SuffixArray();

 private:
  // Constructs the suffix array using the algorithm of Larsson and Sadakane
  // (1999).
  void BuildSuffixArray();

  // Bucket sort on the data array (used for initializing the construction of
  // the suffix array.)
  void InitialBucketSort(vector<int>& groups);

  void TernaryQuicksort(int left, int right, int step, vector<int>& groups);

  // Constructs the suffix array in log(n) steps by doubling the length of the
  // suffixes at each step.
  void PrefixDoublingSort(vector<int>& groups);

  // Given a [low, high) range in the suffix array in which all elements have
  // the first offset-1 values the same, it returns the first position where the
  // offset value is greater or equal to word_id.
  int LookupRangeStart(int low, int high, int word_id, int offset) const;

  shared_ptr<DataArray> data_array;
  vector<int> suffix_array;
  vector<int> word_start;
};

} // namespace extractor

#endif

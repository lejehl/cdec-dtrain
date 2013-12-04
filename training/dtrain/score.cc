
#include "score.h"

namespace dtrain
{


/*
 * bleu
 *
 * as in "BLEU: a Method for Automatic Evaluation
 *        of Machine Translation"
 * (Papineni et al. '02)
 *
 * NOTE: 0 if for one n \in {1..N} count is 0
 */
score_t
BleuScorer::Bleu(NgramCounts& counts, const unsigned hyp_len, const unsigned ref_len)
{
  if (hyp_len == 0 || ref_len == 0) return 0.;
  unsigned M = N_;
  vector<score_t> v = w_;
  if (ref_len < N_) {
    M = ref_len;
    for (unsigned i = 0; i < M; i++) v[i] = 1/((score_t)M);
  }
  score_t sum = 0;
  for (unsigned i = 0; i < M; i++) {
    if (counts.sum_[i] == 0 || counts.clipped_[i] == 0) return 0.;
    sum += v[i] * log((score_t)counts.clipped_[i]/counts.sum_[i]);
  }
  return brevity_penalty(hyp_len, ref_len) * exp(sum);
}

score_t
BleuScorer::Score(const vector<WordID>& hyp, const vector<WordID>& ref,
                  const unsigned /*rank*/, const unsigned /*src_len*/)
{
  unsigned hyp_len = hyp.size(), ref_len = ref.size();
  if (hyp_len == 0 || ref_len == 0) return 0.;
  NgramCounts counts = make_ngram_counts(hyp, ref, N_);
  return Bleu(counts, hyp_len, ref_len);
}

/*
 * 'stupid' bleu
 *
 * as in "ORANGE: a Method for Evaluating
 *        Automatic Evaluation Metrics
 *        for Machine Translation"
 * (Lin & Och '04)
 *
 * NOTE: 0 iff no 1gram match ('grounded')
 */
score_t
StupidBleuScorer::Score(const vector<WordID>& hyp, const vector<WordID>& ref,
                        const unsigned /*rank*/, const unsigned /*src_len*/)
{
  unsigned hyp_len = hyp.size(), ref_len = ref.size();
  if (hyp_len == 0 || ref_len == 0) return 0.;
  NgramCounts counts = make_ngram_counts(hyp, ref, N_);
  unsigned M = N_;
  vector<score_t> v = w_;
  if (ref_len < N_) {
    M = ref_len;
    for (unsigned i = 0; i < M; i++) v[i] = 1/((score_t)M);
  }
  score_t sum = 0, add = 0;
  for (unsigned i = 0; i < M; i++) {
    if (i == 0 && (counts.sum_[i] == 0 || counts.clipped_[i] == 0)) return 0.;
    if (i == 1) add = 1;
    sum += v[i] * log(((score_t)counts.clipped_[i] + add)/((counts.sum_[i] + add)));
  }
  return  brevity_penalty(hyp_len, ref_len) * exp(sum);
}

/*
 * fixed 'stupid' bleu
 *
 * as in "Optimizing for Sentence-Level BLEU+1
 *        Yields Short Translations"
 * (Nakov et al. '12)
 */
score_t
FixedStupidBleuScorer::Score(const vector<WordID>& hyp, const vector<WordID>& ref,
                        const unsigned /*rank*/, const unsigned /*src_len*/)
{
  unsigned hyp_len = hyp.size(), ref_len = ref.size();
  if (hyp_len == 0 || ref_len == 0) return 0.;
  NgramCounts counts = make_ngram_counts(hyp, ref, N_);
  unsigned M = N_;
  vector<score_t> v = w_;
  if (ref_len < N_) {
    M = ref_len;
    for (unsigned i = 0; i < M; i++) v[i] = 1/((score_t)M);
  }
  score_t sum = 0, add = 0;
  for (unsigned i = 0; i < M; i++) {
    if (i == 0 && (counts.sum_[i] == 0 || counts.clipped_[i] == 0)) return 0.;
    if (i == 1) add = 1;
    sum += v[i] * log(((score_t)counts.clipped_[i] + add)/((counts.sum_[i] + add)));
  }
  return  brevity_penalty(hyp_len, ref_len+1) * exp(sum); // <- fix
}

/*
 * smooth bleu
 *
 * as in "An End-to-End Discriminative Approach
 *        to Machine Translation"
 * (Liang et al. '06)
 *
 * NOTE: max is 0.9375 (with N=4)
 */
score_t
SmoothBleuScorer::Score(const vector<WordID>& hyp, const vector<WordID>& ref,
                        const unsigned /*rank*/, const unsigned /*src_len*/)
{
  unsigned hyp_len = hyp.size(), ref_len = ref.size();
  if (hyp_len == 0 || ref_len == 0) return 0.;
  NgramCounts counts = make_ngram_counts(hyp, ref, N_);
  unsigned M = N_;
  if (ref_len < N_) M = ref_len;
  score_t sum = 0.;
  vector<score_t> i_bleu;
  for (unsigned i = 0; i < M; i++) i_bleu.push_back(0.);
  for (unsigned i = 0; i < M; i++) {
    if (counts.sum_[i] == 0 || counts.clipped_[i] == 0) {
      break;
    } else {
      score_t i_ng = log((score_t)counts.clipped_[i]/counts.sum_[i]);
      for (unsigned j = i; j < M; j++) {
        i_bleu[j] += (1/((score_t)j+1)) * i_ng;
      }
    }
    sum += exp(i_bleu[i])/pow(2.0, (double)(N_-i));
  }
  return brevity_penalty(hyp_len, ref_len) * sum;
}

/*
 * 'sum' bleu
 *
 * sum up Ngram precisions
 */
score_t
SumBleuScorer::Score(const vector<WordID>& hyp, const vector<WordID>& ref,
                        const unsigned /*rank*/, const unsigned /*src_len*/)
{
  unsigned hyp_len = hyp.size(), ref_len = ref.size();
  if (hyp_len == 0 || ref_len == 0) return 0.;
  NgramCounts counts = make_ngram_counts(hyp, ref, N_);
  unsigned M = N_;
  if (ref_len < N_) M = ref_len;
  score_t sum = 0.;
  unsigned j = 1;
  for (unsigned i = 0; i < M; i++) {
    if (counts.sum_[i] == 0 || counts.clipped_[i] == 0) break;
    sum += ((score_t)counts.clipped_[i]/counts.sum_[i])/pow(2.0, (double) (N_-j+1));
    j++;
  }
  return brevity_penalty(hyp_len, ref_len) * sum;
}

/*
 * 'sum' (exp) bleu
 *
 * sum up exp(Ngram precisions)
 */
score_t
SumExpBleuScorer::Score(const vector<WordID>& hyp, const vector<WordID>& ref,
                        const unsigned /*rank*/, const unsigned /*src_len*/)
{
  unsigned hyp_len = hyp.size(), ref_len = ref.size();
  if (hyp_len == 0 || ref_len == 0) return 0.;
  NgramCounts counts = make_ngram_counts(hyp, ref, N_);
  unsigned M = N_;
  if (ref_len < N_) M = ref_len;
  score_t sum = 0.;
  unsigned j = 1;
  for (unsigned i = 0; i < M; i++) {
    if (counts.sum_[i] == 0 || counts.clipped_[i] == 0) break;
    sum += exp(((score_t)counts.clipped_[i]/counts.sum_[i]))/pow(2.0, (double) (N_-j+1));
    j++;
  }
  return brevity_penalty(hyp_len, ref_len) * sum;
}

/*
 * 'sum' (whatever) bleu
 *
 * sum up exp(weight * log(Ngram precisions))
 */
score_t
SumWhateverBleuScorer::Score(const vector<WordID>& hyp, const vector<WordID>& ref,
                        const unsigned /*rank*/, const unsigned /*src_len*/)
{
  unsigned hyp_len = hyp.size(), ref_len = ref.size();
  if (hyp_len == 0 || ref_len == 0) return 0.;
  NgramCounts counts = make_ngram_counts(hyp, ref, N_);
  unsigned M = N_;
  vector<score_t> v = w_;
  if (ref_len < N_) {
    M = ref_len;
    for (unsigned i = 0; i < M; i++) v[i] = 1/((score_t)M);
  }
  score_t sum = 0.;
  unsigned j = 1;
  for (unsigned i = 0; i < M; i++) {
    if (counts.sum_[i] == 0 || counts.clipped_[i] == 0) break;
    sum += exp(v[i] * log(((score_t)counts.clipped_[i]/counts.sum_[i])))/pow(2.0, (double) (N_-j+1));
    j++;
  }
  return brevity_penalty(hyp_len, ref_len) * sum;
}

/*
 * approx. bleu
 *
 * as in "Online Large-Margin Training of Syntactic
 *        and Structural Translation Features"
 * (Chiang et al. '08)
 *
 * NOTE: Needs some more code in dtrain.cc .
 *       No scaling by src len.
 */
score_t
ApproxBleuScorer::Score(const vector<WordID>& hyp, const vector<WordID>& ref,
                        const unsigned rank, const unsigned src_len)
{
  unsigned hyp_len = hyp.size(), ref_len = ref.size();
  if (ref_len == 0) return 0.;
  score_t score = 0.;
  NgramCounts counts(N_);
  if (hyp_len > 0) {
    counts = make_ngram_counts(hyp, ref, N_);
    NgramCounts tmp = glob_onebest_counts_ + counts;
    score = Bleu(tmp, hyp_len, ref_len);
  }
  if (rank == 0) { // 'context of 1best translations'
    glob_onebest_counts_ += counts;
    glob_onebest_counts_ *= discount_;
    glob_hyp_len_ = discount_ * (glob_hyp_len_ + hyp_len);
    glob_ref_len_ = discount_ * (glob_ref_len_ + ref_len);
    glob_src_len_ = discount_ * (glob_src_len_ + src_len);
  }
  return score;
}

/*
 * Linear (Corpus) Bleu
 *
 * as in "Lattice Minimum Bayes-Risk Decoding
 *        for Statistical Machine Translation"
 * (Tromble et al. '08)
 *
 */
score_t
LinearBleuScorer::Score(const vector<WordID>& hyp, const vector<WordID>& ref,
                        const unsigned rank, const unsigned /*src_len*/)
{
  unsigned hyp_len = hyp.size(), ref_len = ref.size();
  if (ref_len == 0) return 0.;
  unsigned M = N_;
  if (ref_len < N_) M = ref_len;
  NgramCounts counts(M);
  if (hyp_len > 0)
    counts = make_ngram_counts(hyp, ref, M);
  score_t ret = 0.;
  for (unsigned i = 0; i < M; i++) {
    if (counts.sum_[i] == 0 || onebest_counts_.sum_[i] == 0) break;
    ret += counts.sum_[i]/onebest_counts_.sum_[i];
  }
  ret = -(hyp_len/(score_t)onebest_len_) + (1./M) * ret;
  if (rank == 0) {
    onebest_len_ += hyp_len;
    onebest_counts_ += counts;
  }
  return ret;
}


MapScorer::MapScorer( string& query_file, string& doc_file, string& relevance_file, string& sw_file, unsigned heap_size, string scoring  )
: docs_( doc_file ), queries_( query_file, relevance_file, sw_file ), retrieval_( scoring, heap_size )
{
  cout << "Created new MapScorer. " << endl
      << "scoring: " << scoring << " heap size: " << heap_size << endl;
  iteration_ = 0; // keep track of where we are in input
  isFirstEpoch_ = true; // obsolete?
  heap_size_= heap_size; // max num retrieved
  batch_size_ = 0; // number of sentences per query
  docs_.loadDocs();
  queries_.loadQueries( docs_ );
}

/*
 * MapScorer
 *
 * uses retrieval score of a query containing the current translation
 * to optimise translations for retrieval
 */
score_t MapScorer::Score( const vector<WordID>& hyp, const vector<WordID>& ref,
    const unsigned rank, const unsigned /*src_len*/ ){
  // generate query
  map<string, Query >::iterator qIter = queries_.getQuery( iteration_ );
  if ( rank == 0 ) {
  cerr << "\n==============================="<< endl << "Sentence "<< iteration_ <<
      " qid: " << qIter->second.doc_id_ << endl;
  qIter->second.printRelDocs();
  }
  qIter->second.clear();
  set<WordID> query;
  qIter->second.setTerms( batch_size_, hyp, queries_.stopwords_, query );

  // run retrieval
  MyHeap results( heap_size_ );
  retrieval_.runRetrieval( query, docs_, qIter->second.document_sample_, results );
//  retrieval_.runRetrieval( query, qIter->second.document_sample_, results );

  // score
  double score = retrieval_.evaluateRetrieval( qIter->second.relevant_docs_ , results );

  return (score_t) score;
}

} // namespace


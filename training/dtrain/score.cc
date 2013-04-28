
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
BleuScorer::Score(vector<WordID>& hyp, vector<WordID>& ref,
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
StupidBleuScorer::Score(vector<WordID>& hyp, vector<WordID>& ref,
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
FixedStupidBleuScorer::Score(vector<WordID>& hyp, vector<WordID>& ref,
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
SmoothBleuScorer::Score(vector<WordID>& hyp, vector<WordID>& ref,
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
SumBleuScorer::Score(vector<WordID>& hyp, vector<WordID>& ref,
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
SumExpBleuScorer::Score(vector<WordID>& hyp, vector<WordID>& ref,
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
SumWhateverBleuScorer::Score(vector<WordID>& hyp, vector<WordID>& ref,
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
ApproxBleuScorer::Score(vector<WordID>& hyp, vector<WordID>& ref,
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
LinearBleuScorer::Score(vector<WordID>& hyp, vector<WordID>& ref,
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

/*
 * MapScorer
 *
 * uses Average Precision of a query containing the current translation
 * optimize translations for retrieval
 */
MapScorer::MapScorer( string query_file, string doc_file, string relevance_file )
: docs_( doc_file ), queries_( query_file, relevance_file )
{
	// load document collection, query collection.
	iteration_ = 0;
	isFirstEpoch_ = true;
	docs_.loadDocs();
	queries_.loadQueries();
}
//

score_t MapScorer::Score( vector<WordID>& hyp, vector<WordID>& ref,
		const unsigned rank, const unsigned /*src_len*/ )
{
	if ( isFirstEpoch_ == true ) {
		return 0.0;
	} else {
	//	TODO: modify query
	map<string, Query >::iterator qIter = queries_.getQuery( iteration_ );
	qIter->second.setTerms(iteration_, hyp );
	MyHeap results( 100 ); //TODO: this should be a parameter
	retrieval( docs_, qIter->second.terms_, results );
	return averagePrecision( results, qIter->second );
	}
}

void MapScorer::retrieval( DocumentCollection& docs, set<WordID>& query, MyHeap& results )
{
	// TODO: heap size should be a parameter!

	// TODO: maybe collection should implement its own iterator?
	for ( map<string, Document>::iterator docIter = docs.collection_.begin();
			docIter != docs.collection_.end(); ++docIter ){
		double score = 0.0;
		// TODO: query should have a term iterator?
		for ( set<WordID>::iterator it = query.begin(); it != query.end(); ++it ){
			score += docIter->second.getScoreForQueryTerm( *it );
		}
		// add to heap if score is greate than 0
		if (score != 0.0){
		pair<string, double> p = make_pair( docIter->first, score );
		results.addPair( p );

		}

	}
}






score_t MapScorer::averagePrecision( MyHeap& results,
    		Query& query )
{
	score_t avPrec = 0.0;

	vector<unsigned> gold( results.size_ );
	vector<unsigned> retrieved( results.size_ );
	// create gold standard
	unsigned counter = 0;
	for ( map<string, unsigned>::iterator i = query.relevant_docs_.begin(); i != query.relevant_docs_.end(); ++i ){
		gold.at( counter ) =  i->second ;
		counter += 1;
	}
	// create results
	for ( unsigned i =0; i < results.heap_.size(); i++ ){
		string docid = results.heap_.at(i).first;
		if ( query.relevant_docs_.count(docid) == 1 ){
			retrieved.at(i) =  query.relevant_docs_[docid] ;
		}
	}

	counter = 0;
	double sum = 0.0;
	// precision at i
	for ( unsigned i = 0; i < gold.size(); i++ ){
		if ( retrieved.at(i) >= gold.at(i) ){
			counter++;
			sum += counter / ( i+1 );
		}
	}

	// normalize by number of relevant docs
	avPrec = sum/query.relevant_docs_.size();
	return avPrec;
}

// in first epoch, fill the sentences_ map of the queries
void MapScorer::addDecodedSrc( vector<WordID>& sent ){
	queries_.setSentence( iteration_, sent );

}

} // namespace


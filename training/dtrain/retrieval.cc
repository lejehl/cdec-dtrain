/*
 * retrieval.cc
 *
 *  Created on: May 10, 2013
 *      Author: laura
 */

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "retrieval.h"
#include "collection.h"
#include "document.h"
#include "wordid.h"
#include "myheap.h"

using namespace std;
/*
 * average Precision, according to LETOR paper
 * Liu et al. (SIGIR'07): LETOR: Benchmark Dataset for Research on
 * Learning to Rank for Information Retrieval
 * This does not distinguish between relevance levels
 */
double RetrievalEval::averagePrecision( unsigned num_rels, vector<unsigned>& retrieved ){
  unsigned counter = 0;
  double sum = 0.0;

  // average precision at i
  for ( unsigned i = 0; i < retrieved.size(); i++ ){
    if ( retrieved.at(i) > 0 ){
      counter++;
      sum += (double) counter / (double)( i+1 );
    }
    //    cout << retrieved.at(i) << " ";
  }
  // normalize by total number of RELEVANT docs
  double avPrec;
  avPrec = sum/ (double) num_rels;
  return avPrec;
}

/*
 * implementation after irbook
 * parameters:
 * vector containing relevance scores for retrieved docs (e.g. 0,1,0,2,2,3,0 ...)
 * vector containing gold standard relevances (e.g. 3,3,2,2,2,1 )
 *
 */
double RetrievalEval::ndcg( vector<unsigned>& retrieved, vector<unsigned>& gold ){
  // calculate normalizing factor Z, so that perfect ranking gets score of 1
  // 1/(sum_j=1_num-rel (2^Rel(j) - 1) / log_2 (j+1 ) )
  double Z;
  double Z_den = 0.0;

  unsigned s;
  if (gold.size() > retrieved.size() )
    s = retrieved.size();
  else s = gold.size();
  for (unsigned i = 0; i< s; i++ ) {
    Z_den += ( pow( 2.0, (double) gold[i] ) - 1  ) / log2 (i+2); // +2 because i starts from 0
  }
  Z = 1/Z_den;

  //calculate dcg for retrieved docs
  //sum_j=1_num-retrieved (2^Rel(j) - 1) / log_2 (j+1 )
  double dcg = 0.0;
  for (unsigned i = 0; i< retrieved.size(); i++ ) {
    if ( retrieved[i] > 0 )
      dcg += ( pow( 2.0, (double) retrieved[i] ) - 1  ) / log2 (i+2); // +2 because i starts from 0
    //    cerr << retrieved.at(i) << " ";
  }

  //  cerr << "\nndcg: " << Z*dcg << endl;
  return Z*dcg;
}

Retrieval::Retrieval( string scoring, unsigned heap_size )
: eval_(){
  heap_size_ = heap_size;
  scoring_ = scoring;
}


void Retrieval::runRetrieval( set<WordID>& query, DocumentCollection& docs, vector<string>& doc_sample, MyHeap& results  ){
  //  for ( map<string, Document>::iterator docIter = docs.collection_.begin();
  //      docIter != docs.collection_.end(); ++docIter ){
  for (unsigned i = 0; i< doc_sample.size(); i++ ) {
    map<string, Document>::iterator docIter = docs.collection_.find(doc_sample[i]);
    double score = 0.0;
    for ( set<WordID>::iterator it = query.begin(); it != query.end(); ++it ){
      score += docIter->second.getScoreForQueryTerm( *it );
    }
    if (score != 0.0){
      pair<string, double> p = make_pair( docIter->first, score );
      results.addPair( p );
    }
  }
}

void Retrieval::runRetrieval( set<WordID>& query, vector< Document*>& docs, MyHeap& results  ){
  //  for ( map<string, Document>::iterator docIter = docs.collection_.begin();
  //      docIter != docs.collection_.end(); ++docIter ){

  for ( unsigned i=0; i<docs.size();i++ ) {
    double score = 0.0;
    for ( set<WordID>::iterator it = query.begin(); it != query.end(); ++it ){
      score += docs[i]->getScoreForQueryTerm( *it );
    }
    if (score != 0.0){
      pair<string, double> p = make_pair( docs[i]->doc_id_, score );
      results.addPair( p );
    }
  }
}

double Retrieval::evaluateRetrieval( map<string, unsigned>& rels, MyHeap& results ){
  double score = 0.0;
  if ( results.heap_.size() == 0 ) return score;
  std::vector< std::pair<string, double>  > r_heap;
  results.reverseHeap( r_heap );

  // get relevances for retrieved docs
  vector<unsigned> retrieved( heap_size_ );
  for ( unsigned i =0; i < retrieved.size(); i++ ){
    try {
      string docid = r_heap.at(i).first;
      if ( rels.count(docid) == 1 ){
        retrieved.at(i) =  rels[docid] ;
      }
    } catch ( const out_of_range& oor ) {
      break;
    }
  }

  // get gold standard relevances
  vector<unsigned> gold;
  getSortedRelevances( gold, rels );

  // calculate ndcg
  score = eval_.ndcg( retrieved, gold );
  return score;
}

void Retrieval::getSortedRelevances( vector<unsigned>& gold, map<string, unsigned>& rels ){
  for ( map<string, unsigned>::iterator iter = rels.begin(); iter != rels.end(); ++iter){
    gold.push_back( iter->second );
  }
  sort( gold.begin(), gold.end(), std::greater<unsigned>());
}



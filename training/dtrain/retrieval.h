/*
 * retrieval.h
 *
 *  Created on: May 10, 2013
 *      Author: laura
 */

#ifndef RETRIEVAL_H_
#define RETRIEVAL_H_

#include <vector>

#include "wordid.h"
#include "collection.h"
#include "myheap.h"


using namespace std;



struct RetrievalEval {
  RetrievalEval() {};
  double ndcg( vector<unsigned>& retrieved, vector<unsigned>& gold );
  double averagePrecision( unsigned num_rels, vector<unsigned>& retrieved );
};

struct Retrieval {
  Retrieval( string scoring = "map",  unsigned heap_size = 10 );
   unsigned heap_size_;
  string scoring_;
   void runRetrieval( set<WordID>& query, DocumentCollection& docs, vector<string>& doc_sample, MyHeap& results );
   void runRetrieval( set<WordID>& query, vector< Document*>& docs, MyHeap& results  );
  double evaluateRetrieval( map<string, unsigned>& rels, MyHeap& result_list  );
  private:
    RetrievalEval eval_;
    void getSortedRelevances( vector<unsigned>& gold, map<string, unsigned>& rels  );
};

#endif /* RETRIEVAL_H_ */

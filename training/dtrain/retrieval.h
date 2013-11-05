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

//	double avPrecAtN( vector<unsigned>& gold, vector<unsigned>& retrieved, unsigned n = 0 );
//private:
	double averagePrecision( unsigned num_rels, vector<unsigned>& retrieved  );
	double ndcg( vector<unsigned>& retrieved, vector<unsigned>&gold );

};

//TODO
struct Retrieval {
	Retrieval( string scoring = "map",  unsigned heap_size = 10 );
 	unsigned heap_size_;
	string scoring_;
 	void runRetrieval( set<WordID>& query, DocumentCollection& docs, MyHeap& results );
	double evaluateRetrieval( map<string, unsigned>& rels, MyHeap& result_list  );
	private:
		RetrievalEval eval_;
//		void retrieval( set<WordID>& query, DocumentCollection& docs, MyHeap& results );
//		double averagePrecision( vector<unsigned>& gold, vector<unsigned>& retrieved );
//		double ndcg( vector<unsigned>& retrieved );
		void getSortedRelevances( vector<unsigned>& gold, map<string, unsigned>& rels  );
};

#endif /* RETRIEVAL_H_ */

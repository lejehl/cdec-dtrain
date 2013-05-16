/*
 * retrieval.cc
 *
 *  Created on: May 10, 2013
 *      Author: laura
 */

#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "retrieval.h"
#include "collection.h"
#include "document.h"
#include "wordid.h"
#include "myheap.h"

using namespace std;

double RetrievalEval::averagePrecision( vector<unsigned>& gold, vector<unsigned>& retrieved ){
	unsigned counter = 0;
	double sum = 0.0;

	// precision at i
	cout << "relevance (gold)\tsum(precision@rank)" << endl;
	for ( unsigned i = 0; i < gold.size(); i++ ){
		if ( retrieved.at(i) >= gold.at(i) ){
			counter++;
			sum += (double) counter / (double)( i+1 );
		}
		cout << retrieved.at(i) << " (" << gold.at(i) << ") " ;
	}
	cout << "\t" << sum;
	double avPrec;

	// normalize by number of RELEVANT docs
	avPrec = sum/gold.size();
//	cout << "\t" << avPrec;
	cout << endl;
	return avPrec;

}

double RetrievalEval::avPrecAtN( vector<unsigned>& gold, vector<unsigned>& retrieved, unsigned n ){
	double avPrec = 0.0;
	if  (  n > retrieved.size()  )  {
		cerr << "n must be less than or equal to num_retrieved!" << endl;
		exit(1);
	} else if ( n == 0 ) {
		n = gold.size();
	} else {
		if (n < gold.size() ) {
				cout << "retrieved has " << retrieved.size() << " elements." << endl;
				cout << "gold has " << gold.size() << " elements." << endl;
				gold.erase( gold.begin()+n, gold.end() );
		}
		if ( n > gold.size() ) {
			for ( unsigned i = gold.size(); i < n; i++ ){
				gold.push_back( 0 );
			}
		}
	}
	avPrec = averagePrecision( gold, retrieved );

	return avPrec;
}

Retrieval::Retrieval( string scoring, unsigned heap_size ){
	heap_size_ = heap_size;
	scoring_ = scoring;
	cout << "created new instance of Retrieval" << endl <<
			"scoring: " << scoring_ << " heap size: " << heap_size_ << endl;
}

void Retrieval::runRetrieval( set<WordID>& query, DocumentCollection& docs, MyHeap& results  ){
	cout << "running retrieval" << endl;
	for ( map<string, Document>::iterator docIter = docs.collection_.begin();
			docIter != docs.collection_.end(); ++docIter ){
		double score = 0.0;
		// TODO: use OpenMP for this
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

double Retrieval::evaluateRetrieval( map<string, unsigned>& rels, MyHeap& results ){
	double score = 0.0;
	vector<unsigned> retrieved( heap_size_ );
	for ( unsigned i =0; i < retrieved.size(); i++ ){
		try {
			string docid = results.heap_.at(i).first;
			if ( rels.count(docid) == 1 ){
				retrieved.at(i) =  rels[docid] ;
			}
		} catch ( const out_of_range& oor ) {
			cerr << "This shouldn't happen!" << endl;
		}
		cout << retrieved.at(i) << " " ;
	}
	if (scoring_ == "map"){
		vector<unsigned> gold;
		getSortedRelevances( gold, rels );
		score =  averagePrecision(  gold, retrieved );
	} else {
		score = ndcg( retrieved );
	}
	cout << "\t" << score << endl;
	return score;
}

void Retrieval::getSortedRelevances( vector<unsigned>& gold, map<string, unsigned>& rels ){
	for ( map<string, unsigned>::iterator iter = rels.begin(); iter != rels.end(); ++iter){
		gold.push_back( iter->second );
	}
	sort( gold.begin(), gold.end(), std::greater<unsigned>());
}

/*
 * Average Precision
 */
double Retrieval::averagePrecision( vector<unsigned>& gold, vector<unsigned>& retrieved ){
	unsigned counter = 0;
	double sum = 0.0;

	// precision at i
	cout << "relevance (gold)\tsum(precision@rank)" << endl;
	for ( unsigned i = 0; i < gold.size(); i++ ){
		if ( retrieved.at(i) >= gold.at(i) ){
			counter++;
			sum += (double) counter / (double)( i+1 );
		}
		cout << retrieved.at(i) << " (" << gold.at(i) << ") " ;
	}
	cout << "\t" << sum;
	double avPrec;

	// normalize by number of RELEVANT docs
	avPrec = sum/gold.size();
//	cout << "\t" << avPrec;
	cout << endl;
	return avPrec;
}

/*
 * normalized discounted cumulative gain
 */
double Retrieval::ndcg( vector<unsigned>& retrieved ){
	cout << "Scoring: ndcg" << endl;
	double dcg = (double) retrieved[0];
	vector<double> normalizers;
	normalizers.push_back( 1.0 );
	for (unsigned i=2; i <= retrieved.size(); i++){
		// log(base 2) of rank
		double normalizer = log ( i ) / log ( 2 );
		dcg += (double) retrieved[i-1] / normalizer;
		// remember normalizer for dcg
		normalizers.push_back( normalizer );
	}
	sort ( retrieved.begin(), retrieved.end(), std::greater<unsigned>());
	double idcg = 0.0;
	for (unsigned i=1; i <= retrieved.size(); i++){
		idcg += (double) retrieved[i-1] / normalizers[i-1];
	}
	return dcg/idcg;
}

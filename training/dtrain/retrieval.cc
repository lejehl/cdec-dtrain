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
//	cout << "relevance (gold)\tsum(precision@rank)" << endl;
	for ( unsigned i = 0; i < retrieved.size(); i++ ){
		if ( retrieved.at(i) > 0 ){
			counter++;
			sum += (double) counter / (double)( i+1 );
		}
		cout << retrieved.at(i) << " ";
	}
	double avPrec;
	// normalize by total number of RELEVANT docs
	avPrec = sum/ (double) num_rels;
//	cout << endl;
//	cout << "average Precision: " << avPrec << endl;
//	cout << "retrieved relevant docs: " << counter << endl;
//	cout << "total relevant docs: " << num_rels << endl;
	cout << avPrec << endl;
	return avPrec;
}

/*
 * implementation after irbook
 */
double RetrievalEval::ndcg( vector<unsigned>& retrieved, vector<unsigned>& gold ){


	// calculate normalizing factor, so that perfect ranking gets score of 1
	// 1/(sum_j=1_num-rel (2^Rel(j) - 1) / log_2 (j+1 ) )

	double Z;
	double Z_den = 0.0;
	// go over relevant docs, since quotient is 0 if relevance is 0
//	assert( gold.size() <= retrieved.size() );
	unsigned s;
	if (gold.size() > retrieved.size() )
		s = retrieved.size();
	else s = gold.size();
	for (unsigned i = 0; i< s; i++ ) {
		Z_den += ( pow( 2.0, (double) gold[i] ) - 1  ) / log2 (i+2); // +2 because i starts from 0
//		cout << "Z_den: " << Z_den << " ";
	}
	Z = 1/Z_den;
//	cout << " Z: " << Z << endl;

    //calculate dcg for retrieved docs
	//sum_j=1_num-retrieved (2^Rel(j) - 1) / log_2 (j+1 )
	double dcg = 0.0;
	for (unsigned i = 0; i< retrieved.size(); i++ ) {
		if ( retrieved[i] > 0 )
		dcg += ( pow( 2.0, (double) retrieved[i] ) - 1  ) / log2 (i+2); // +2 because i starts from 0
//		cout << "dcg: " << dcg << " ";
		cout << retrieved.at(i) << " ";
	}
	cout << "\nndcg: " << Z*dcg << endl;
	return Z*dcg;

//	cout << "Scoring: ndcg" << endl;
//	double dcg = (double) retrieved[0];
//	vector<double> normalizers;
//	normalizers.push_back( 1.0 );
//	for (unsigned i=2; i <= retrieved.size(); i++){
//		// log(base 2) of rank
//		double normalizer = log ( i ) / log ( 2 );
//		dcg += (double) retrieved[i-1] / normalizer;
//		cout << "dcg@" << i << ": " << retrieved[i-1] / normalizer << endl;
//		// remember normalizer for dcg
//		normalizers.push_back( normalizer );
//	}
//	cout << "total dcg: " << dcg << endl;
//	sort ( retrieved.begin(), retrieved.end(), std::greater<unsigned>());
//	double idcg = 0.0;
//	for (unsigned i=1; i <= retrieved.size(); i++){
//		idcg += (double) retrieved[i-1] / normalizers[i-1];
//	}
//	cout << "total idcg: " << idcg << endl;
//	cout << "ndcg: " << dcg/idcg << endl;
//	return dcg/idcg;
}


//double RetrievalEval::avPrecAtN( vector<unsigned>& gold, vector<unsigned>& retrieved, unsigned n ){
//	double avPrec = 0.0;
//	if  (  n > retrieved.size()  )  {
//		cerr << "n must be less than or equal to num_retrieved!" << endl;
//		exit(1);
//	} else if ( n == 0 ) {
//		n = gold.size();
//	} else {
//		if (n < gold.size() ) {
//				cout << "retrieved has " << retrieved.size() << " elements." << endl;
//				cout << "gold has " << gold.size() << " elements." << endl;
//				gold.erase( gold.begin()+n, gold.end() );
//		}
//		if ( n > gold.size() ) {
//			for ( unsigned i = gold.size(); i < n; i++ ){
//				gold.push_back( 0 );
//			}
//		}
//	}
//	avPrec = averagePrecision( gold, retrieved );
//
//	return avPrec;
//}

Retrieval::Retrieval( string scoring, unsigned heap_size )
: eval_(){
	heap_size_ = heap_size;
	scoring_ = scoring;
	cout << "created new instance of Retrieval" << endl <<
			"scoring: " << scoring_ << " heap size: " << heap_size_ << endl;
}

void Retrieval::runRetrieval( set<WordID>& query, DocumentCollection& docs, MyHeap& results  ){
	cout << "Terms in query: " << query.size() << endl;
	for ( map<string, Document>::iterator docIter = docs.collection_.begin();
			docIter != docs.collection_.end(); ++docIter ){
		double score = 0.0;
		// TODO: use OpenMP for this
		for ( set<WordID>::iterator it = query.begin(); it != query.end(); ++it ){
			score += docIter->second.getScoreForQueryTerm( *it );
		}
		// add to heap if score is greater than 0
		if (score != 0.0){
			pair<string, double> p = make_pair( docIter->first, score );
			results.addPair( p );
		}
	}
//	results.printHeap();
	cout <<  results.heap_.size() << " documents returned" << endl;

}

double Retrieval::evaluateRetrieval( map<string, unsigned>& rels, MyHeap& results ){
	double score = 0.0;
	if ( results.heap_.size() == 0 ) return score;
	std::vector< std::pair<string, double>  > r_heap;
    results.reverseHeap( r_heap );

	vector<unsigned> retrieved( heap_size_ );
	for ( unsigned i =0; i < retrieved.size(); i++ ){
		try {
			string docid = r_heap.at(i).first;
			if ( rels.count(docid) == 1 ){
//				cout << "document " << docid << " was found." << endl;
				retrieved.at(i) =  rels[docid] ;
			}
		} catch ( const out_of_range& oor ) {
//			cerr << "Heap is smaller than " << "retrieved.size( "
//					<< heap_size_ << " )" << endl;
			break;
		}
	}
	if (scoring_ == "map"){
//		score =  eval_.averagePrecision(  rels.size(), retrieved );
//	} else {
		vector<unsigned> gold;
		getSortedRelevances( gold, rels );
		score = eval_.ndcg( retrieved, gold );
	}
	return score;
}

void Retrieval::getSortedRelevances( vector<unsigned>& gold, map<string, unsigned>& rels ){
	for ( map<string, unsigned>::iterator iter = rels.begin(); iter != rels.end(); ++iter){
		gold.push_back( iter->second );
	}
	sort( gold.begin(), gold.end(), std::greater<unsigned>());
}



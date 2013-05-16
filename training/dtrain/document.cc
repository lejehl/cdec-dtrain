/*
 * document.cc
 *
 *  Created on: Apr 5, 2013
 *      Author: laura
 */

#include <math.h>
#include <iostream>
#include <algorithm>
#include "document.h"


using namespace std;

// make an empty tf vector and assign doc id and size
TextItem::TextItem( const string& docid ): tf_vector_()
{
  doc_id_ = docid;
  doc_size_ = 0;
// cout << "Made new TextItem!" << endl;
}

void TextItem::generateTfVector( vector<WordID>& text )
{
	doc_size_ = 0;
	for( unsigned int i=0; i<text.size(); i++ ){
			tf_vector_[text.at(i)] += 1;
			doc_size_ += 1;
		}

}




Document::Document( const string& docid)
: TextItem( docid ), weighted_vector_()
{
//	cerr << " created new Document: " << docid <<  endl;
	}


void Document::generateBM25Vector( map<WordID, unsigned>& dftable,
		double avg_len, double num_docs )
{
	double a = avg_len;
	double n = num_docs;
	for ( map<WordID,unsigned>::iterator iter=tf_vector_.begin(); iter != tf_vector_.end(); ++ iter)
	{
				WordID term = iter->first;
				double tf = iter-> second;
				double df = dftable[ iter-> first];
				double bm25 = BM25( tf, df, a, n );
				weighted_vector_[ term ] = bm25;
			}

}



double Document::getScoreForQueryTerm( WordID s )
{

	map<WordID, double>::iterator pos = weighted_vector_.find(s);
	if ( pos != weighted_vector_.end() ){
		return pos->second;
	}
	else return 0.0;
}

double Document::BM25( double tf, double df, double avg_len, double num_docs,
		double k , double b )
{
	double score = 0.0;
	double numerator = tf * ( k + 1);
	double denominator = k * ((1-b) + b * ( doc_size_ / avg_len ) + tf + tf );
	double idf = log( (num_docs - df + 0.5 )/( df + 0.5 ) );
	score = idf * ( numerator / denominator );
	return score;
}

Query::Query( const string& docid )
: TextItem( docid ), relevant_docs_(), sentences_(), terms_()
{
//	cerr << " created new Query: " << docid <<  endl;
}

void Query::setRelevantDocs( string& docid, unsigned relscore )
{
	relevant_docs_[docid] = relscore;
}

void Query::setSentence( unsigned sent_id, vector<WordID>& text )
{
	sentences_[sent_id] = text;
}


void Query::setTerms( unsigned sentId, vector<WordID>& text ){
	setSentence( sentId, text );
	for (map<unsigned, vector<WordID> >::iterator it=sentences_.begin();
			it != sentences_.end(); ++it  ){
		for ( unsigned i=0; i < it->second.size(); i++ ){
			terms_.insert( it->second.at(i) );
		}
	}
}

void Query::setTerms( ){
	for (map<unsigned, vector<WordID> >::iterator it=sentences_.begin();
			it != sentences_.end(); ++it  ){
		for ( unsigned i=0; i < it->second.size(); i++ ){
			terms_.insert( it->second.at(i) );
		}
	}
}


void Query::printRelDocs(){
	cout << "relevant documents for query " << doc_id_ << ": ";
	for ( map<string, unsigned>::iterator iter = relevant_docs_.begin(); iter != relevant_docs_.end(); ++iter){
		cout << iter->first << "	" << iter->second <<  endl;
	}
}

vector<unsigned> Query::getSortedRelevances( ){

//	cout << "returning sorted vector" << endl;
	vector<unsigned> rels;
	for ( map<string, unsigned>::iterator iter = relevant_docs_.begin(); iter != relevant_docs_.end(); ++iter){
		rels.push_back( iter->second );
//		cout << iter->second << endl;
	}
	sort( rels.begin(), rels.end(), std::greater<unsigned>());
	return rels;
}




/*
 * document.cc
 *
 *  Created on: Apr 5, 2013
 *      Author: laura
 */

#include <math.h>
#include <iostream>

#include "document.h"


using namespace std;

// make an empty tf vector and assign doc id and size
TextItem::TextItem( const string& docid ): tf_vector_()
{
  doc_id_ = docid;
  doc_size_ = 0;
  cout << "Made new TextItem!" << endl;
}

// inline?
void TextItem::generateTfVector( vector<WordID>& text )
{
	doc_size_ = 0;
	for( unsigned int i=0; i<text.size(); i++ ){
			tf_vector_[text.at(i)] += 1;
			doc_size_ += 1;
		}

}

//debug
//void TextItem::printTfVector()
//{
//	for ( map<string,unsigned>::iterator iter=tf_vector_.begin(); iter != tf_vector_.end(); ++ iter)
//	{
//		cout << iter->first << "=>" << iter->second << ", ";
//	}
//	cout << endl;
//
//}



Document::Document( const string& docid)
: TextItem( docid ), weighted_vector_()
{}


void Document::generateBM25Vector( map<WordID, unsigned>& dftable,
		double avg_len, double num_docs )
{
	double a = avg_len;
	double n = num_docs;
	for ( map<WordID,unsigned>::iterator iter=tf_vector_.begin(); iter != tf_vector_.end(); ++ iter)
	{
				// better use a pointer here???
				WordID term = iter->first;
				double tf = iter-> second;
				double df = dftable[ iter-> first];
				double bm25 = BM25( tf, df, a, n );
				weighted_vector_[ term ] = bm25;
			}

}

//void Document::printWeightedVector()
//{
//	for ( map<string,double>::iterator iter=weighted_vector_.begin(); iter != weighted_vector_.end(); ++ iter)
//	{
//		cout << iter->first << "=>" << iter->second << ", ";
//	}
//	cout << endl;
//
//}

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
{}

void Query::setRelevantDocs( string& docid, unsigned relscore )
{
	relevant_docs_[docid] = relscore;
}

void Query::setSentence( unsigned sent_id, vector<WordID>& text )
{
	sentences_[sent_id] = text;
}

// TODO: is this being used?
//vector<string> Query::getSentences()
//{
//	vector<string> s;
//	for (map<unsigned, vector<string> >::iterator it=sentences_.begin(); it != sentences_.end(); ++it  )
//	{
//		s.insert( s.end(), it->second.begin(), it->second.end() );
//	}
//	return s;
//}
//

// TODO: is this being used?
//vector<string> Query::getSentences( unsigned sent_id, vector<string>& new_sentence )
//{
//	vector<string> s;
//	for (map<unsigned, vector<string> >::iterator it=sentences_.begin(); it != sentences_.end(); ++it  )
//	{
//		if (it->first != sent_id)
//		{
//			s.insert( s.end(), it->second.begin(), it->second.end() );
//		}
//		else
//			s.insert( s.end(), new_sentence.begin(), new_sentence.end() );
//	}
//	return s;
//}

void Query::setTerms( unsigned sentId, vector<WordID>& text ){
	setSentence( sentId, text );
	for (map<unsigned, vector<WordID> >::iterator it=sentences_.begin();
			it != sentences_.end(); ++it  ){
		for ( unsigned i=0; i < it->second.size(); i++ ){
			terms_.insert( it->second.at(i) );
		}
	}
}

// use vectors instead?



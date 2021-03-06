/*
 * document.h
 *
 *  Created on: Apr 5, 2013
 *      Author: laura
 */

#ifndef DOCUMENT_H_
#define DOCUMENT_H_

#include<map>
#include<vector>
#include<string>
#include<set>

#include "wordid.h"
#include "tdict.h"

using namespace std;

struct TextItem
{
public:
  TextItem( const string& /*docID*/ );
  string doc_id_;
  unsigned doc_size_;
  map<WordID, unsigned> tf_vector_;
  void generateTfVector( vector<WordID>& text  );
};

struct Document : public TextItem
{
public:
  Document( const string& docid = "" );
  map<WordID, double> weighted_vector_;
  void generateBM25Vector( map<WordID, unsigned>& dftable, double avg_len, double num_docs );
  double getScoreForQueryTerm ( WordID );
private:
  double BM25( double tf, double df, double avg_len, double num_docs, double k = 1.2, double b = 0.75);

};

struct Query : public TextItem
{
public:
  Query( const string& docid = ""  );
  map<string,unsigned> relevant_docs_;
  map<unsigned, vector<WordID> > sentences_; // query word tokens
  set<WordID> terms_; // query word types
//  vector< Document* > document_sample_; // all relevant and sample of 200 irrelevant docs for this query
  vector< string > document_sample_;
  void setRelevantDocs( string& docid, unsigned relscore );
  void setTerms( unsigned sent_id, const vector<WordID>& text, const set<WordID>& sw, set<WordID>& q  );
  void setTerms();
  void setSentence( unsigned sent_id, vector<WordID>& text  );
  void printRelDocs( );
  vector<unsigned> getSortedRelevances();
  void clear();
};


#endif /* DOCUMENT_H_ */

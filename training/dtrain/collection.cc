/*
 * collection.cc
 *
 *  Created on: Apr 4, 2013
 *      Author: laura
 */

#include "collection.h"
#include "dtrain.h"

using namespace std;
using namespace dtrain;


Collection::Collection( const string& filename )//: collection_()
{
	infile_ = filename;
	num_docs_ = 0;
	cout << "Made new collection: " <<  "filename: " << infile_
			<< ", num docs: " << num_docs_ << endl;
}



unsigned Collection::getNumDocs()
{
	return num_docs_;
}

void Collection::splitOnTabs( string& in, vector<string>& parts )
{
	stringstream instream(in);
	string column;
	while ( getline( instream, column, '\t' ) ){
		parts.push_back( column );
	}
}

void Collection::tokenize( string& text, vector<string>& text_tok )
{
	boost::split(text_tok, text, boost::is_any_of(" "));
}


DocumentCollection::DocumentCollection( string& filename )
: Collection( filename ), collection_(), dftable_()
{
	avg_len_ = 0.0;
//	cout << "filename " << infile_ << endl;
	// TODO: optimise for larger document collections
	loadDocs();
	cout << "Have " << num_docs_ << " documents." << endl;
}


void DocumentCollection::generateDfTable( )
{
	for (map<string,Document>::iterator iter2=collection_.begin(); iter2 != collection_.end(); ++iter2)
	{
		map<WordID,unsigned> tfVec = iter2->second.tf_vector_;
		for ( map<WordID,unsigned>::iterator tf_iter=tfVec.begin(); tf_iter != tfVec.end(); ++tf_iter)
		{
				dftable_[tf_iter->first] += 1;
				}
	}

}

/*
 * for each document-id, populate the document collection.
 * calculate the dftable, bm25 weights and average document length
 */
void DocumentCollection::loadDocs(  )
{
	cerr << " loading documents..." << endl;
	ReadFile input( infile_ );
	string in;

	// make sure num_docs_ is 0
	num_docs_ = 0;
	while (getline( *input, in)) {
		vector<string> parts;
		splitOnTabs( in, parts );
		string docid = parts[0];
		string text = parts[1];

		//tokenize
		vector<string> text_tok;
		tokenize( text, text_tok );

		//create Document and generate tf vector
		vector<WordID> word_id_vec;
		register_and_convert( text_tok, word_id_vec );
		collection_[ docid ] = Document( docid );
		map<string, Document>::iterator Doc = collection_.find(docid);
		Doc->second.generateTfVector( word_id_vec );

		// increase num_docs
		num_docs_ += 1;
		}

	averageDocLength();

	// make dftable from tfvectors
	cerr << " making dftable..." << endl;
	generateDfTable( );

   // make bm25 weighted vectors from dftable and tfvectors
	cerr << " making BM25 vectors ... " << endl;
	for (map<string,Document> ::iterator iter=collection_.begin(); iter != collection_.end(); ++iter)
			{
				double a = getAvgLen();
				double n = getNumDocs();
				iter->second.generateBM25Vector( dftable_, a, n);
			}
//			cerr << " finished." << endl << "num docs: " << num_docs_ << endl
//			<< "avg len: " << avg_len_ << endl << "collection size: " << collection_.size() << endl;


}

void DocumentCollection::averageDocLength()
{
	double total = 0.0;
	for ( map<string,Document>::iterator iter=collection_.begin(); iter != collection_.end(); ++iter){
		double l = double(iter->second.doc_size_);
		total += l;
//		cout << "total" << total << endl;
	}
	avg_len_ = total /double( num_docs_ ) ;
}

double DocumentCollection::getAvgLen()
{
	return avg_len_;
}

QueryCollection::QueryCollection( string& filename, string& relevance_file )
: Collection( filename ), collection_(), sentence_qid_map_()
{
	relfile_ = relevance_file;
	loadQueries();
}

/*
 * for each query-id, load the relevance judgments and populate the sentence-qid-map.
 * Don't load the query text, since it needs to be translated first.
 */
void QueryCollection::loadQueries()
{
	cerr << " loading queries" << endl;
	cout << "qid\tnum_rels" << endl;
	ReadFile input( infile_ );
	ReadFile rels( relfile_ );
	string in;
	string rel_record;
	string prev_qid = "";
	string qid = "";
	unsigned num_sentences = 0;

	while (getline( *input, in )) {
		vector<string> parts;
		splitOnTabs( in, parts );
		string qid = parts[0];
		if ( qid != prev_qid ) {
			// create new query in collection
			collection_[ qid ] = Query(qid);
			map<string, Query>::iterator Q = collection_.find(qid);
			cout << qid << "\t";
			while(getline( *rels, rel_record )){
				vector<string> rel_parts;
				splitOnTabs( rel_record, rel_parts );
				string rel_qid = rel_parts[0];
				string rel_doc = rel_parts[2];
				unsigned rel_score;
				stringstream(rel_parts[3]) >> rel_score ;
				if ( rel_qid !=  qid ){
					break;
				}
				else {
					Q->second.setRelevantDocs( rel_doc, rel_score );
				}


			}
			cout << Q->second.relevant_docs_.size() << endl;
//			Q->second.printRelDocs();
			num_docs_ += 1;
			prev_qid = qid;
		}
		sentence_qid_map_[num_sentences] = qid;
		num_sentences += 1;


	}

//	cout << "Finished Loading!" << endl << "filename: " << infile_ << endl
//			 << endl << "num docs: " << num_docs_ << endl
//			<< "collection size: " << collection_.size() << endl;

}


void QueryCollection::setSentence( unsigned sentid, vector<WordID> & text_tok )
{
	string qid = sentence_qid_map_[ sentid ];
	Query * qPtr;
	qPtr = & collection_[qid];// TODO: is this a pointer to the query with ID qid?
	qPtr->setSentence( sentid, text_tok);


}


map<string, Query >::iterator QueryCollection::getQuery( unsigned sent_id )
{
//	cout << "Finding query for sentence " << sent_id << endl;
	string qid = sentence_qid_map_[ sent_id ];
//	cout << "Found query ID: " << qid << endl;
	return collection_.find( qid );

}

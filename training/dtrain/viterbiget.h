/*
 * viterbiget.h
 *
 *  Created on: 6 May 2013
 *      Author: jehl
 */

#ifndef VITERBIGET_H_
#define VITERBIGET_H_

#include "viterbi.h"
#include "dtrain.h"

namespace dtrain
{

struct ViterbiGetter : public HypSampler
{

public:
  vector<WordID> transl_;

  ViterbiGetter():transl_() {};
  virtual void
    NotifyTranslationForest(const SentenceMetadata& smeta, Hypergraph* hg)
    {

      transl_ = getViterbiTranslation(*hg);
    }

  vector<WordID> getViterbiTranslation( const Hypergraph& forest ){
    vector<WordID> result;
    ViterbiESentence( forest, &result );
    return result;

  }


  // this will never be used
  vector<ScoredHyp>* GetSamples() {
    vector<ScoredHyp>* v;
    return v;  };
};

}


#endif /* VITERBIGET_H_ */

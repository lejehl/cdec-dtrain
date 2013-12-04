#include <omp.h>
#include "dtrain.h"
#include "score.h"
#include "kbestget.h"
#include "ksampler.h"
#include "pairsampling.h"
#include "viterbiget.h"
#include "logval.h"


using namespace dtrain;


bool
dtrain_init(int argc, char** argv, po::variables_map* cfg)
{
  po::options_description ini("Configuration File Options");
  ini.add_options()
                                                                            ("input",             po::value<string>()->default_value("-"),                                             "input file (src)")
                                                                            ("refs,r",            po::value<string>(),                                                                       "references")
                                                                            ("output",            po::value<string>()->default_value("-"),                          "output weights file, '-' for STDOUT")
                                                                            ("input_weights",     po::value<string>(),                                "input weights file (e.g. from previous iteration)")
                                                                            ("decoder_config",    po::value<string>(),                                                      "configuration file for cdec")
                                                                            ("print_weights",     po::value<string>(),                                               "weights to print on each iteration")
                                                                            ("stop_after",        po::value<unsigned>()->default_value(0),                                 "stop after X input sentences")
                                                                            ("keep",              po::value<bool>()->zero_tokens(),                               "keep weights files for each iteration")
                                                                            ("epochs",            po::value<unsigned>()->default_value(10),                               "# of iterations T (per shard)")
                                                                            ("k",                 po::value<unsigned>()->default_value(100),                            "how many translations to sample")
                                                                            ("sample_from",       po::value<string>()->default_value("kbest"),     "where to sample translations from: 'kbest', 'forest'")
                                                                            ("filter",            po::value<string>()->default_value("uniq"),                          "filter kbest list: 'not', 'uniq'")
                                                                            ("pair_sampling",     po::value<string>()->default_value("XYX"),                 "how to sample pairs: 'all', 'XYX' or 'PRO'")
                                                                            ("hi_lo",             po::value<float>()->default_value(0.1),                   "hi and lo (X) for XYX (default 0.1), <= 0.5")
                                                                            ("pair_threshold",    po::value<score_t>()->default_value(0.),                         "bleu [0,1] threshold to filter pairs")
                                                                            ("N",                 po::value<unsigned>()->default_value(4),                                          "N for Ngrams (BLEU)")
                                                                            ("scorer",            po::value<string>()->default_value("stupid_bleu"), "scoring: bleu, stupid_, smooth_, approx_, lc_, map")
                                                                            ("learning_rate",     po::value<weight_t>()->default_value(1.0),                                              "learning rate")
                                                                            ("gamma",             po::value<weight_t>()->default_value(0.),                            "gamma for SVM (0 for perceptron)")
                                                                            ("select_weights",    po::value<string>()->default_value("last"),     "output best, last, avg weights ('VOID' to throw away)")
                                                                            ("rescale",           po::value<bool>()->zero_tokens(),                              "rescale weight vector after each input")
                                                                            ("l1_reg",            po::value<string>()->default_value("none"), "apply l1 regularization as in 'Tsuroka et al' (2010) UNTESTED")
                                                                            ("l1_reg_strength",   po::value<weight_t>(),                                                     "l1 regularization strength")
                                                                            ("fselect",           po::value<weight_t>()->default_value(-1), "select top x percent (or by threshold) of features after each epoch NOT IMPLEMENTED") // TODO
                                                                            ("approx_bleu_d",     po::value<score_t>()->default_value(0.9),                                   "discount for approx. BLEU")
                                                                            ("scale_bleu_diff",   po::value<bool>()->zero_tokens(),                      "learning rate <- bleu diff of a misranked pair")
                                                                            ("loss_margin",       po::value<weight_t>()->default_value(0.),  "update if no error in pref pair but model scores this near")
                                                                            ("max_pairs",         po::value<unsigned>()->default_value(std::numeric_limits<unsigned>::max()), "max. # of pairs per Sent.")
                                                                            ("noup",              po::value<bool>()->zero_tokens(),                                               "do not update weights")
                                                                            // additional options for scoring with MAP
                                                                            ("query_file", 		  po::value<string>()->default_value(""),										"mapscoring: query file" )
                                                                            ("doc_file",		  po::value<string>()->default_value(""),								"mapscoring: document collection")
                                                                            ("rel_file",		  po::value<string>()->default_value(""),							  "mapscoring: relevance annotations")
                                                                            ("sw_file",		      po::value<string>()->default_value(""),							  	"mapscoring: stopword file (optional)")
                                                                            ("num_retrieved",	  po::value<unsigned>()->default_value(10),					  "mapscoring: number of retrieved documents")
                                                                            ("num_threads",	  	  po::value<unsigned>()->default_value(4),					  "mapscoring: number of parallel threads")
                                                                            ("query_trans",	  	  po::value<string>()->default_value(""),					  "mapscoring: supply initial query translations")
                                                                            ("iters_per_epoch",	  po::value<unsigned>()->default_value(500),				   "CLIR-scoring: number of training iterations per epoch")
                                                                            ("online_update",	  po::value<bool>()->default_value(false),				   "CLIR-scoring: update weights after every training instance (pair)?")
                                                                            ("avg_perceptron",	  po::value<bool>()->default_value(false),				   "CLIR-scoring: use Collins-style averaged weights (Collins, EMNLP '02) - ONLY WORKS WITH ONLINE UPDATE!");


  po::options_description cl("Command Line Options");
  cl.add_options()
                                                                            ("config,c",         po::value<string>(),              "dtrain config file")
                                                                            ("quiet,q",          po::value<bool>()->zero_tokens(),           "be quiet")
                                                                            ("verbose,v",        po::value<bool>()->zero_tokens(),         "be verbose");
  cl.add(ini);
  po::store(parse_command_line(argc, argv, cl), *cfg);
  if (cfg->count("config")) {
    ifstream ini_f((*cfg)["config"].as<string>().c_str());
    po::store(po::parse_config_file(ini_f, ini), *cfg);
  }
  po::notify(*cfg);
  if (!cfg->count("decoder_config")) {
    cerr << cl << endl;
    return false;
  }
  if ((*cfg)["sample_from"].as<string>() != "kbest"
      && (*cfg)["sample_from"].as<string>() != "forest") {
    cerr << "Wrong 'sample_from' param: '" << (*cfg)["sample_from"].as<string>() << "', use 'kbest' or 'forest'." << endl;
    return false;
  }
  if ((*cfg)["sample_from"].as<string>() == "kbest" && (*cfg)["filter"].as<string>() != "uniq" &&
      (*cfg)["filter"].as<string>() != "not") {
    cerr << "Wrong 'filter' param: '" << (*cfg)["filter"].as<string>() << "', use 'uniq' or 'not'." << endl;
    return false;
  }
  if ((*cfg)["pair_sampling"].as<string>() != "all" && (*cfg)["pair_sampling"].as<string>() != "XYX" &&
      (*cfg)["pair_sampling"].as<string>() != "PRO") {
    cerr << "Wrong 'pair_sampling' param: '" << (*cfg)["pair_sampling"].as<string>() << "'." << endl;
    return false;
  }
  if(cfg->count("hi_lo") && (*cfg)["pair_sampling"].as<string>() != "XYX") {
    cerr << "Warning: hi_lo only works with pair_sampling XYX." << endl;
  }
  if((*cfg)["hi_lo"].as<float>() > 0.5 || (*cfg)["hi_lo"].as<float>() < 0.01) {
    cerr << "hi_lo must lie in [0.01, 0.5]" << endl;
    return false;
  }
  if ((*cfg)["pair_threshold"].as<score_t>() < 0) {
    cerr << "The threshold must be >= 0!" << endl;
    return false;
  }
  if ((*cfg)["select_weights"].as<string>() != "last" && (*cfg)["select_weights"].as<string>() != "best" &&
      (*cfg)["select_weights"].as<string>() != "avg" && (*cfg)["select_weights"].as<string>() != "VOID") {
    cerr << "Wrong 'select_weights' param: '" << (*cfg)["select_weights"].as<string>() << "', use 'last' or 'best'." << endl;
    return false;
  }
  return true;
}

int
main(int argc, char** argv)
{
  // handle most parameters

  po::variables_map cfg;
  if (!dtrain_init(argc, argv, &cfg)) exit(1); // something is wrong
  bool quiet = false;
  if (cfg.count("quiet")) quiet = true;
  bool verbose = false;
  if (cfg.count("verbose")) verbose = true;
  bool noup = false;
  if (cfg.count("noup")) noup = true;
  bool rescale = false;
  if (cfg.count("rescale")) rescale = true;
  bool keep = false;
  if (cfg.count("keep")) keep = true;

  const unsigned k = cfg["k"].as<unsigned>();
  const unsigned N = cfg["N"].as<unsigned>();
  const unsigned T = cfg["epochs"].as<unsigned>();
  const unsigned stop_after = cfg["stop_after"].as<unsigned>();
  const string filter_type = cfg["filter"].as<string>();
  const string sample_from = cfg["sample_from"].as<string>();
  const string pair_sampling = cfg["pair_sampling"].as<string>();
  const score_t pair_threshold = cfg["pair_threshold"].as<score_t>();
  const string select_weights = cfg["select_weights"].as<string>();
  const float hi_lo = cfg["hi_lo"].as<float>();
  const score_t approx_bleu_d = cfg["approx_bleu_d"].as<score_t>();
  const unsigned max_pairs = cfg["max_pairs"].as<unsigned>();
  weight_t loss_margin = cfg["loss_margin"].as<weight_t>();
  if (loss_margin > 9998.) loss_margin = std::numeric_limits<float>::max();
  bool scale_bleu_diff = false;
  if (cfg.count("scale_bleu_diff")) scale_bleu_diff = true;
  bool average = false;
  if (select_weights == "avg")
    average = true;
  vector<string> print_weights;
  if (cfg.count("print_weights"))
    boost::split(print_weights, cfg["print_weights"].as<string>(), boost::is_any_of(" "));


  // setup decoder
  register_feature_functions();
  SetSilent(true);
  ReadFile ini_rf(cfg["decoder_config"].as<string>());
  if (!quiet)
    cerr << setw(25) << "cdec cfg " << "'" << cfg["decoder_config"].as<string>() << "'" << endl;
  Decoder decoder(ini_rf.stream());

  // scoring metric/scorer
  string scorer_str = cfg["scorer"].as<string>();

  // only for map score: read queries and docs
  string query_fn =  cfg["query_file"].as<string>();
  string doc_fn =  cfg["doc_file"].as<string>();
  string rels_fn =  cfg["rel_file"].as<string>();
  string stopwords =  cfg["sw_file"].as<string>();
  unsigned num_retr = cfg["num_retrieved"].as<unsigned>();
  unsigned threads = cfg["num_threads"].as<unsigned>();
  unsigned num_iters = cfg["iters_per_epoch"].as<unsigned>();
  bool online_update = cfg["online_update"].as<bool>();
  bool avg_perceptron = cfg["avg_perceptron"].as<bool>();
  string init_trans_fn =  cfg["query_trans"].as<string>();
  ReadFile init_trans;
  if ( init_trans_fn != "" ) {
    init_trans.Init(  init_trans_fn );
  }


#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
#endif

  cout << "num retrieved is " << num_retr << endl;

  cerr << "creating scorer ..." << endl;
  LocalScorer* scorer;
  if (scorer_str == "bleu") {
    scorer = static_cast<BleuScorer*>(new BleuScorer);
  } else if (scorer_str == "stupid_bleu") {
    scorer = static_cast<StupidBleuScorer*>(new StupidBleuScorer);
  } else if (scorer_str == "fixed_stupid_bleu") {
    scorer = static_cast<FixedStupidBleuScorer*>(new FixedStupidBleuScorer);
  } else if (scorer_str == "smooth_bleu") {
    scorer = static_cast<SmoothBleuScorer*>(new SmoothBleuScorer);
  } else if (scorer_str == "sum_bleu") {
    scorer = static_cast<SumBleuScorer*>(new SumBleuScorer);
  } else if (scorer_str == "sumexp_bleu") {
    scorer = static_cast<SumExpBleuScorer*>(new SumExpBleuScorer);
  } else if (scorer_str == "sumwhatever_bleu") {
    scorer = static_cast<SumWhateverBleuScorer*>(new SumWhateverBleuScorer);
  } else if (scorer_str == "approx_bleu") {
    scorer = static_cast<ApproxBleuScorer*>(new ApproxBleuScorer(N, approx_bleu_d));
  } else if (scorer_str == "lc_bleu") {
    scorer = static_cast<LinearBleuScorer*>(new LinearBleuScorer(N));
  } else if (scorer_str == "map" ) {
    scorer = static_cast<MapScorer*>(new MapScorer( query_fn, doc_fn, rels_fn, stopwords , num_retr ));
  } else {
    cerr << "Don't know scoring metric: '" << scorer_str << "', exiting." << endl;
    exit(1);
  }
  vector<score_t> bleu_weights;
  scorer->Init(N, bleu_weights);
  cerr << "done" << endl;

  // setup decoder observer
  MT19937 rng; // random number generator, only for forest sampling
  HypSampler* observer;
  if (sample_from == "kbest")
    observer = static_cast<KBestGetter*>(new KBestGetter(k, filter_type));
  else
    observer = static_cast<KSampler*>(new KSampler(k, &rng));
  observer->SetScorer(scorer);


  // init weights
  cerr << "initialize weights... " << endl;
  vector<weight_t>& dense_weights = decoder.CurrentWeightVector();
  SparseVector<weight_t> lambdas, cumulative_penalties, w_average;
  if (cfg.count("input_weights")) Weights::InitFromFile(cfg["input_weights"].as<string>(), &dense_weights);
  Weights::InitSparseVector(dense_weights, &lambdas);

  // meta params for perceptron, SVM
  cerr  << "set meta parameters ... " << endl;
  weight_t eta = cfg["learning_rate"].as<weight_t>();
  weight_t gamma = cfg["gamma"].as<weight_t>();

  // faster perceptron: consider only misranked pairs, see
  // DO NOT ENABLE  WITH SVM (gamma > 0) OR loss_margin!
  bool faster_perceptron = false;
  if (gamma==0 && loss_margin==0) faster_perceptron = true;

  // l1 regularization
  cerr << "set regularization params ... " << endl;
  bool l1naive = false;
  bool l1clip = false;
  bool l1cumul = false;
  weight_t l1_reg = 0;
  if (cfg["l1_reg"].as<string>() != "none") {
    string s = cfg["l1_reg"].as<string>();
    if (s == "naive") l1naive = true;
    else if (s == "clip") l1clip = true;
    else if (s == "cumul") l1cumul = true;
    l1_reg = cfg["l1_reg_strength"].as<weight_t>();
  }

  // output
  cerr << "set input and output ... " << endl;
  string output_fn = cfg["output"].as<string>();
  // input
  string input_fn = cfg["input"].as<string>();
  ReadFile input(input_fn);
  // buffer input for t > 0
  cerr << "buffer input and read refs ... " << endl;
  vector<string> src_str_buf;          // source strings (decoder takes only strings)
  vector<vector<WordID> > ref_ids_buf; // references as WordID vecs
  string refs_fn = cfg["refs"].as<string>();
  ReadFile refs(refs_fn);


  unsigned in_sz = std::numeric_limits<unsigned>::max(); // input index, input size
  vector<pair<score_t, score_t> > all_scores;
  score_t max_score = 0.;
  unsigned best_it = 0;
  float overall_time = 0.;

  // output cfg
  if (!quiet) {
    cerr << _p5;
    cerr << endl << "dtrain" << endl << "Parameters:" << endl;
    cerr << setw(25) << "k " << k << endl;
    cerr << setw(25) << "N " << N << endl;
    cerr << setw(25) << "T " << T << endl;
    cerr << setw(26) << "scorer '" << scorer_str << "'" << endl;
    if (scorer_str == "approx_bleu")
      cerr << setw(25) << "approx. B discount " << approx_bleu_d << endl;
    cerr << setw(25) << "sample from " << "'" << sample_from << "'" << endl;
    if (sample_from == "kbest")
      cerr << setw(25) << "filter " << "'" << filter_type << "'" << endl;
    if (!scale_bleu_diff) cerr << setw(25) << "learning rate " << eta << endl;
    else cerr << setw(25) << "learning rate " << "bleu diff" << endl;
    cerr << setw(25) << "gamma " << gamma << endl;
    cerr << setw(25) << "loss margin " << loss_margin << endl;
    cerr << setw(25) << "faster perceptron " << faster_perceptron << endl;
    cerr << setw(25) << "pairs " << "'" << pair_sampling << "'" << endl;
    if (pair_sampling == "XYX")
      cerr << setw(25) << "hi lo " << hi_lo << endl;
    cerr << setw(25) << "pair threshold " << pair_threshold << endl;
    cerr << setw(25) << "select weights " << "'" << select_weights << "'" << endl;
    if (cfg.count("l1_reg"))
      cerr << setw(25) << "l1 reg " << l1_reg << " '" << cfg["l1_reg"].as<string>() << "'" << endl;
    if (rescale)
      cerr << setw(25) << "rescale " << rescale << endl;
    cerr << setw(25) << "max pairs " << max_pairs << endl;
    cerr << setw(25) << "cdec cfg " << "'" << cfg["decoder_config"].as<string>() << "'" << endl;
    cerr << setw(25) << "input " << "'" << input_fn << "'" << endl;
    cerr << setw(25) << "refs " << "'" << refs_fn << "'" << endl;
    cerr << setw(25) << "output " << "'" << output_fn << "'" << endl;
    if (cfg.count("input_weights"))
      cerr << setw(25) << "weights in " << "'" << cfg["input_weights"].as<string>() << "'" << endl;
    if (stop_after > 0)
      cerr << setw(25) << "stop_after " << stop_after << endl;
    if (!verbose) cerr << "(a dot represents " << DTRAIN_DOTS << " inputs)" << endl;
  }
  cerr << "finished setup .. " << endl;

  //  if MAP is used, do one pass over the data and set the viterbi translations in the scorer. why do we need this?
  if ( scorer_str == "map" ) {

    ViterbiGetter* v = new ViterbiGetter();
    string in;
    string trans;
    vector< vector<WordID> > hyps;
    unsigned it = 0;
    WriteFile qt;
    if (init_trans_fn == "" ) {
      qt.Init("queries.transl"); // works with '-'
    }
    ostream& o = *qt.stream();

    while(getline(*input, in)){
      //		  if ( ii+1 % 5 == 0 ){
      if (init_trans_fn == "" ) {
        if (it == 0)
          cerr << "setting viterbi translations" << endl;
        cerr << "." ;
        if ( (it+1) % 30 == 0 )
          cerr << it+1 << endl;
        decoder.Decode( in, v);
        hyps.push_back( v->transl_ );
        for (unsigned i = 0; i< v->transl_.size(); i++ ) {
          o << TD::Convert( v->transl_[i]);
          //        cerr << TD::Convert( v->transl_[i]);
          if (i < v->transl_.size()-1) o << " ";
        }
        o << endl;
      } else {
        if (it == 0 )
          cerr << "loading translations" << endl;
        getline(*init_trans, trans );
        vector<string> tok;
        vector<WordID> ids;
        boost::split( tok, trans, boost::is_any_of(" "));
        register_and_convert( tok, ids);
        hyps.push_back( ids );
      }
      src_str_buf.push_back( in );
      it++;
      scorer->increaseIter();
    }

    scorer->updateSentences(hyps );
    in_sz = it;
    scorer->Reset();
    cerr << endl;
    cerr << "done" << endl;
    delete v;
  } // end first input loop

  //	if ("scorer_str" == "map" ) {
  SparseVector<weight_t> diff_vec; // initialize weight vector (for mini-batch update)
  unsigned batch_size = 0;
  vector< vector<WordID> > top_hyps;
  WriteFile lf("loss"); // works with '-'
  ostream& o = *lf.stream();
  o.precision(17);

  // begin outer loop
  for (unsigned t = 0; t < T; t++) // T epochs
  {
    time_t start, end, start_opt, end_opt;
    time(&start);
    score_t score_sum = 0.;
    score_t model_sum(0);
    int converged_after = num_iters; // for averaged perceptron
    SparseVector<weight_t> sum_lambdas; // for averaged perceptron
    unsigned ii = 0, rank_errors = 0, margin_violations = 0, npairs = 0, f_count = 0, list_sz = 0;
    if (!quiet) cerr << "Iteration #" << t+1 << " of " << T << "." << endl;
    score_t loss = 0.;
    vector<pair<ScoredHyp,ScoredHyp> > batch_pairs;

    // begin inner loop
    while(true)
    {
      string in;
      bool next = false, stop = false; // next iteration or premature stop
      if ( (t == 0) && (scorer_str != "map") ) {
        //      if ( t == 0 ) {
        //    	cerr << "this shoulnd't happen if scoring with map" << endl;
        if(!getline(*input, in)) next = true;
        //      cerr << "no lines in input " << endl;
      } else {
        //    	cerr << "ii == sz - this also shouldn't happen" << endl;
        if (ii == in_sz) next = true; // stop if we reach the end of our input
      }
      // stop after X sentences (but still go on for those)
      if (stop_after > 0 && stop_after == ii && !next) stop = true;

      // produce some pretty output
      if (!quiet && !verbose) {
        if (ii == 0) cerr << " ";
        if ((ii+1) % (DTRAIN_DOTS) == 0) {
          cerr << ".";
          cerr.flush();
        }
        if ((ii+1) % (20*DTRAIN_DOTS) == 0) {
          cerr << " " << ii+1 << endl;
          if (!next && !stop) cerr << " ";
        }
        if (stop) {
          if (ii % (20*DTRAIN_DOTS) != 0) cerr << " " << ii << endl;
          cerr << "Stopping after " << stop_after << " input sentences." << endl;
        } else {
          if (next) {
            if (ii % (20*DTRAIN_DOTS) != 0) cerr << " " << ii << endl;
          }
        }
      }

      // next iteration
      if (next || stop) break;

      // weights
      // pass weights to decoder
      lambdas.init_vector(&dense_weights);

      // getting input
      cout << "read references..." << endl;

      vector<WordID> ref_ids; // reference as vector<WordID>
      if (t == 0 && scorer_str != "map") {
        src_str_buf.push_back(in);
      }

      if (scorer_str != "map" )
      {
        if (t == 0 ) {
          string r_;
          getline(*refs, r_);
          vector<string> ref_tok;
          boost::split(ref_tok, r_, boost::is_any_of(" "));
          register_and_convert(ref_tok, ref_ids);
          ref_ids_buf.push_back(ref_ids);
        } else {
          ref_ids = ref_ids_buf[ii];
        }
        observer->SetRef(ref_ids);

      }
      if ( (t == 0) && ( scorer_str != "map") ){
        cout << "decoding...\n";
        decoder.Decode(in, observer);
      } else {
        cout << "decoding...\n";
        decoder.Decode(src_str_buf[ii], observer);
      }

      // get (scored) samples
      cout << "get scored samples..." << endl;
      vector<ScoredHyp>* samples = observer->GetSamples();

      if (verbose) {
        if ( scorer_str != "map") {
          cerr << "--- ref for " << ii << ": ";
          if (t > 0) printWordIDVec(ref_ids_buf[ii]);
          else printWordIDVec(ref_ids);
          cerr << endl;
        }
        for (unsigned u = 0; u < samples->size(); u++) {
          cerr << _p2 << _np << "[" << u << ". '";
          printWordIDVec((*samples)[u].w);
          cerr << "'" << endl;
          cerr << "SCORE=" << (*samples)[u].score << ",model="<< (*samples)[u].model << endl;
          cerr << "F{" << (*samples)[u].f << "} ]" << endl << endl;
        }
      }

      score_sum += (*samples)[0].score; // stats for 1best
      model_sum += (*samples)[0].model;

      f_count += observer->get_f_count();
      list_sz += observer->get_sz();

      // for MAP: increase iteration
      if (scorer_str == "map") {
        int max_pos;
        score_t curr_max = 0;

        // find maximum scoring hypothesis
        for ( int i =0; i < (*samples).size(); i++ ) {
          if ( (*samples)[i].score > curr_max ) {
            max_pos = i;
            curr_max = (*samples)[i].score;
          }
        }
        cerr << "top1 hyp score: " << (*samples)[0].score << endl;
        cerr << "maximum scoring hyp: " << max_pos << ", score: " << curr_max << endl;
        top_hyps.push_back( (*samples)[max_pos].w );
        scorer->increaseIter( );
      }


      // weight updates
      if (!noup  ) {

        // get pairs
        cout << "get pairs ...\n";
        vector<pair<ScoredHyp,ScoredHyp> > pairs;
        if (pair_sampling == "all")
          all_pairs(samples, pairs, pair_threshold, max_pairs, faster_perceptron);
        if (pair_sampling == "XYX")
          partXYX(samples, pairs, pair_threshold, max_pairs, faster_perceptron, hi_lo);
        if (pair_sampling == "PRO")
          PROsampling(samples, pairs, pair_threshold, max_pairs);
        npairs += pairs.size();

        // add pairs to batch
        for (vector<pair<ScoredHyp,ScoredHyp> >::iterator it = pairs.begin();
            it != pairs.end(); it++) {
          batch_pairs.push_back( *it );
        }
      }
      ++ii;

    } // input loop


    // do update for all pairs
    if (!noup  ) {
      time(&start_opt);
      cerr.precision(17);
      const unsigned s = batch_pairs.size();
      cerr << "do " << num_iters << " optimization iterations over " << s << " pairs.\n";
      vector< SparseVector<weight_t> > grads_j;
      grads_j.resize(s);
      vector <score_t> loss_j;
      loss_j.resize(s);
      vector<bool> rc_j;
      rc_j.resize(s);
      for (int i = num_iters; i> 0; i--) {
        loss = 0.;
        loss_j.clear();
        grads_j.clear();
        //        re_j.clear();
        if (i%50 == 0 ) cerr << "." ;

        // BEGIN PARALLEL
        // parallelize gradient aggregation over all pairs
        //# pragma omp parallel for
        //        for (unsigned j = 0; j< s;  j++ ) {
        unsigned j = 0;
        unsigned num_constraints = 0;
        unsigned num_rank_errs = 0;
        for (vector< pair<ScoredHyp, ScoredHyp> >::iterator it=batch_pairs.begin(); it != batch_pairs.end(); ++it ) {
            cerr << _p5 << _p << "WEIGHTS" << endl;
            for (vector<string>::iterator it = print_weights.begin(); it != print_weights.end(); it++) {
              cerr << setw(18) << *it << " = " << lambdas.get(FD::Convert(*it)) << endl;
            }
          if ( rc_j[j] == true ) {
            if ( avg_perceptron ) {
              sum_lambdas += lambdas;
            }
            continue; // if the pair is ranked correctly, don't reconsider it
          }

          bool rank_error;
          //          score_t margin;
          score_t margin = std::numeric_limits<float>::max();
          //          if (faster_perceptron) { // we only have considering misranked pairs
          //            rank_error = true; // pair sampling already did this for us
          //            margin = std::numeric_limits<float>::max();
          //          } else {
          //            rank_error = it->first.model <= it->second.model;

          // update model score
          //          vector< pair<ScoredHyp, ScoredHyp> >::iterator it = batch_pairs.begin()+j;
          if (i < num_iters ) {
            it->first.model = it->first.f.dot(lambdas) ;
            it->second.model = it->second.f.dot(lambdas);
            //            batch_pairs[j].first.model = batch_pairs[j].first.f.dot(lambdas) ;
            //            batch_pairs[j].second.model = batch_pairs[j].second.f.dot(lambdas);
          }


          rank_error = it->first.model <= it->second.model;
          if ( ! rank_error ) {
            margin = fabs((it->first.model) - (it->second.model));
          } else {
            num_rank_errs++;
          }
          //          cerr << "first model: "<< it->first.model << " | second model" << it->second.model << " | rank err? "<< rank_error << endl;


          //            rank_error = it->first.model <= it->second.model;
          //            margin = fabs((it->first.model) - (it->second.model));
          //            if (!rank_error && margin < loss_margin) margin_violations++;

          //          }
          //          if (rank_error) rank_errors++;
          //          if (scale_bleu_diff) eta = it->first.score - it->second.score;

          // calculate gradient if margin < loss_margin
          if (rank_error || margin < loss_margin) {

            //            loss += loss_margin - ( batch_pairs[j].first.model - batch_pairs[j].second.model);
            //            loss_j[j] = loss_margin - ( it->first.model - it->second.model);
            loss += loss_margin - ( it->first.model - it->second.model);

            //              //        		SparseVector<weight_t> diff_vec = it->first.f - it->second.f; // calculate difference between feature vectors (gradient)
            if ( online_update ) {
              diff_vec = it->first.f - it->second.f;
              lambdas.plus_eq_v_times_s(diff_vec, eta);  // add learning rate * gradient, update after every pair
            } else {
              //            }

              //            grads_j[j] = it->first.f - it->second.f;
              diff_vec += it->first.f - it->second.f;
            }
            num_constraints++;

            rc_j[j] = false;
          } else { rc_j[j] = true; }

          if (avg_perceptron)
            sum_lambdas += lambdas;
          //          }

          //            if (gamma)
          //              lambdas.plus_eq_v_times_s(lambdas, -2*gamma*eta*(1./npairs));
          j++;

        }// END PARALLEL


        // calculate batch gradient and batch loss
        //        unsigned num_constraints = 0;
        //        if (scorer_str == "map"  ) {
        //          for (unsigned i =0; i< s; i++) {
        //            if (re_j[i] == true ) {
        //              num_constraints ++;
        //              loss += loss_j[i];
        //              diff_vec += grads_j[i];
        //            }
        //          }
        if (! online_update ) {
          lambdas.plus_eq_v_times_s(diff_vec, (1/(double) num_constraints) * eta);
        }
        diff_vec.clear();
        // stop if there are no ranking errors or margin violations
        if (num_rank_errs== 0 and num_constraints == 0 ) {
          converged_after = num_iters-i+1;
          break;
        }
        o << loss << endl;
        cerr << "num_constraints: " << num_constraints << ", num_rank_errs: " << num_rank_errs << endl;
      }
      cerr << "converged after " << converged_after << " iterations\n";


      // l1 regularization
      // please note that this penalizes _all_ weights
      // (contrary to only the ones changed by the last update)
      // after a _sentence_ (not after each example/pair)
      if (l1naive) {
        FastSparseVector<weight_t>::iterator it = lambdas.begin();
        for (; it != lambdas.end(); ++it) {
          it->second -= sign(it->second) * l1_reg;
        }
      } else if (l1clip) {
        FastSparseVector<weight_t>::iterator it = lambdas.begin();
        for (; it != lambdas.end(); ++it) {
          if (it->second != 0) {
            weight_t v = it->second;
            if (v > 0) {
              it->second = max(0., v - l1_reg);
            } else {
              it->second = min(0., v + l1_reg);
            }
          }
        }
      } else if (l1cumul) {
        weight_t acc_penalty = (ii+1) * l1_reg; // ii is the index of the current input
        FastSparseVector<weight_t>::iterator it = lambdas.begin();
        for (; it != lambdas.end(); ++it) {
          if (it->second != 0) {
            weight_t v = it->second;
            weight_t penalized = 0.;
            if (v > 0) {
              penalized = max(0., v-(acc_penalty + cumulative_penalties.get(it->first)));
            } else {
              penalized = min(0., v+(acc_penalty - cumulative_penalties.get(it->first)));
            }
            it->second = penalized;
            cumulative_penalties.set_value(it->first, cumulative_penalties.get(it->first)+penalized);
          }
        }
      }

      time(&end_opt);
      float opt_time = difftime(end_opt, start_opt);
      cerr << "This took " << opt_time << " seconds" << endl;
      cerr << endl;

    } else {
      cout << "no update.\n";
    }


    if (scorer_str == "map" && top_hyps.size() != 0 && scorer->end_of_batch ) {
      scorer->updateSentences( top_hyps );
      top_hyps.clear();
    }

    if (rescale) lambdas /= lambdas.l2norm();

    //      ++ii;
    //
    //    } // input loop

    if (average)  {
      if ( avg_perceptron ) {
//        w_average.plus_eq_v_times_s(sum_lambdas, 1.0/( converged_after*batch_pairs.size())  );
        lambdas = sum_lambdas * ( 1.0/( converged_after*batch_pairs.size()) ); // use averaged weights for next epoch
        cerr << "Calculating average: sum_lambdas * 1.0/" << converged_after << " * " << batch_pairs.size() << endl;
      } else {
      w_average += lambdas;
      }
    }
    if (scorer_str == "approx_bleu" || scorer_str == "lc_bleu" || scorer_str == "map") scorer->Reset();

    if (t == 0) {
      in_sz = ii; // remember size of input (# lines)
    }

    // print some stats
    score_t score_avg = score_sum/(score_t)in_sz;
    score_t model_avg = model_sum/(score_t)in_sz;
    score_t score_diff, model_diff;
    if (t > 0) {
      score_diff = score_avg - all_scores[t-1].first;
      model_diff = model_avg - all_scores[t-1].second;
    } else {
      score_diff = score_avg;
      model_diff = model_avg;
    }

    unsigned nonz = 0;
    if (!quiet) nonz = (unsigned)lambdas.num_nonzero();

    if (!quiet) {
      cerr << _p5 << _p << "WEIGHTS" << endl;
      for (vector<string>::iterator it = print_weights.begin(); it != print_weights.end(); it++) {
        cerr << setw(18) << *it << " = " << lambdas.get(FD::Convert(*it)) << endl;
      }
      cerr << "        ---" << endl;
      cerr << _np << "       1best avg score: " << score_avg;
      cerr << _p << " (" << score_diff << ")" << endl;
      cerr << _np << " 1best avg model score: " << model_avg;
      cerr << _p << " (" << model_diff << ")" << endl;
      cerr << "           avg # pairs: ";
      cerr << _np << npairs/(float)in_sz;
      if (faster_perceptron) cerr << " (meaningless)";
      cerr << endl;
      cerr << "        avg # rank err: ";
      cerr << rank_errors/(float)in_sz << endl;
      cerr << "     avg # margin viol: ";
      cerr << margin_violations/(float)in_sz << endl;
      cerr << "    non0 feature count: " <<  nonz << endl;
      cerr << "           avg list sz: " << list_sz/(float)in_sz << endl;
      cerr << "           avg f count: " << f_count/(float)list_sz << endl;
      cerr << "           total loss: " << loss << endl;
    }

    pair<score_t,score_t> remember;
    remember.first = score_avg;
    remember.second = model_avg;
    all_scores.push_back(remember);
    if (score_avg > max_score) {
      max_score = score_avg;
      best_it = t;
    }
    time (&end);
    float time_diff = difftime(end, start);
    overall_time += time_diff;
    if (!quiet) {
      cerr << _p2 << _np << "(time " << time_diff/60. << " min, ";
      cerr << time_diff/in_sz << " s/S)" << endl;
    }
    if (t+1 != T && !quiet) cerr << endl;

    if (noup) break;
    // only for MAP
    //  if ( scorer_str == "map" && t == 0 ) continue;

    // write weights to file
    if (select_weights == "best" || keep) {
      lambdas.init_vector(&dense_weights);
      string w_fn = "weights." + boost::lexical_cast<string>(t) + ".gz";
      Weights::WriteToFile(w_fn, dense_weights, true);
    }

    // eta for next iteration: use decreasing learning rate
    //    eta /= 2;
  } // outer loop

  if (average) w_average /= (weight_t)T;

  if (!noup) {
    if (!quiet) cerr << endl << "Writing weights file to '" << output_fn << "' ..." << endl;
    if (select_weights == "last" || average) { // last, average
      WriteFile of(output_fn); // works with '-'
      ostream& o = *of.stream();
      o.precision(17);
      o << _np;
      if (average) {
        for (SparseVector<weight_t>::iterator it = w_average.begin(); it != w_average.end(); ++it) {
          if (it->second == 0) continue;
          o << FD::Convert(it->first) << '\t' << it->second << endl;
        }
      } else {
        for (SparseVector<weight_t>::iterator it = lambdas.begin(); it != lambdas.end(); ++it) {
          if (it->second == 0) continue;
          o << FD::Convert(it->first) << '\t' << it->second << endl;
        }
      }
    } else if (select_weights == "VOID") { // do nothing with the weights
    } else { // best
      if (output_fn != "-") {
        CopyFile("weights."+boost::lexical_cast<string>(best_it)+".gz", output_fn);
      } else {
        ReadFile bestw("weights."+boost::lexical_cast<string>(best_it)+".gz");
        string o;
        cout.precision(17);
        cout << _np;
        while(getline(*bestw, o)) cout << o << endl;
      }
      if (!keep) {
        for (unsigned i = 0; i < T; i++) {
          string s = "weights." + boost::lexical_cast<string>(i) + ".gz";
          unlink(s.c_str());
        }
      }
    }
    if (!quiet) cerr << "done" << endl;
  }

  if (!quiet) {
    cerr << _p5 << _np << endl << "---" << endl << "Best iteration: ";
    cerr << best_it+1 << " [SCORE '" << scorer_str << "'=" << max_score << "]." << endl;
    cerr << "This took " << overall_time/60. << " min." << endl;
  }
}


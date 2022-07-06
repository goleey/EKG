class alp_config:
    def __init__(self, path = "/foursquare_twitter/"):
        self.train_ratio = 0.8
        self.train_neg_ratio = 1
        self.test_neg_ratio = 29

        self.anchors = path + "anchors_all"
        self.train = path + "train_anchors"
        self.test = path + "test_anchors"

        self.x_train = path + "x_train"
        self.y_train = path + "y_train"
        self.x_test = path + "x_test"
        self.y_test = path + "y_test"
        self.test_candidates = path + "test_candidates"

        self.foursquare_checkins = path + "foursquare_checkins"
        self.foursquare_loc_map = path + "foursquare_loc_map"
        self.foursquare_loc_comment = path + "foursquare_loc_comment"
        #extract keyphrase by "Simple Unsupervised Keyphrase Extraction using Sentence Embeddings"
        self.foursquare_loc_keyphrase = path + "foursquare_loc_keyphrase"
        #crawl type information of POI from web
        self.loc2category_foursquare = path + "sup_loc2category_foursquare.json"
        self.sub_loc2category_foursquare = path + "sub_loc2category_foursquare.json"
        self.twitter_uid_tweeets = path + "twitter_uid_tweets"
        # self.twitter_text_lookup = path + "twitter_text_lookup"
        #extract keyphrase by "Simple Unsupervised Keyphrase Extraction using Sentence Embeddings"
        self.twitter_key_phrase = path + "twitter_key_phrase"

        # Data visualization, maybe useless in the future.
        self.info_in_same_day = path + "info_in_same_day"

        self.corps_all = path + "all_corps_file"
        self.glove_50d = path + "glove.6B.50d.txt"

        self.neg_dict = path + "neg_dict"

        self.twitter_poi = path + "twitter_poi.json"
        self.num_doc_class = 10

        self.f_entity_time_pair = path + "f_entity_time_pair.pkl"
        self.t_entity_time_pair = path + "t_entity_time_pair.pkl"

        self.f_entityid_time_pair = path + "f_entityid_time_pair.pkl"
        self.t_entityid_time_pair = path + "t_entityid_time_pair.pkl"

        self.entity_linkage_dic = "./crawl/entity_linkage_all.json"


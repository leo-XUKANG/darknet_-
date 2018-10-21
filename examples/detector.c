#include "darknet.h"
#include "network.h"
static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};


void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);//read the cfg file 
    char *train_images = option_find_str(options, "train", "data/train.list"); //read the image you want to train
    char *backup_directory = option_find_str(options, "backup", "/backup/");//the path that you will save the weights file

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);//this will load the model to the gpus
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0)); //set the seed of the random number
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;//the number of image that you read is the batchsize*subdivisions*ngpus in one time
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer; //the data structure in the darknet.h,yes the structure is predeclared,what you should to do is to use it
 
    layer l = net->layers[net->n - 1];

    int classes = l.classes; //the number of class
    float jitter = l.jitter;

    list *plist = get_paths(train_images); //get the path of the images
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist); //get the path list of the images

    load_args args = get_base_args(net); //i do know what is this used for ,but i do not understand how this work?
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;//it will use threads to make it faster

    pthread_t load_thread = load_data(args); //this function maybe create the thread according to your network args
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){ 
//get_current_batch() will return the current batch times,so if the the current batch time is smaller than the max_batches that you have set,it will continue to train
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;//use the rand() to get a random number
            if (get_current_batch(net)+200 > net->max_batches) dim = 608; //what does it use to do,to make the best result better? i dont understand
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim); //this is not to resize the picture,it is to resize the whole network,it will resize each layer in net ,the memory,the space
            }
            net = nets[0];
        } //maybe this part is for promote the robust of  the network

        time=what_time_is_it_now();//according the time
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);//print how many time used in read data

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train); //training on cpu
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%1000==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
//change by xukang,anyway ,i just change it, not create it ,make it save the weights that get bigger val_acc,
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c*2);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
            if (nms) do_nms_sort(dets, num, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, num, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, num, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR);
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/2007_test.list");
    char *name_list = option_find_str(options, "names", "data/voc.names");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR);
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


typedef struct {
	box b;
	float p;
	int class_id;
	int image_index;
	int truth_flag;
	int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
	box_prob a = *(box_prob *)pa;
	box_prob b = *(box_prob *)pb;
	float diff = a.p - b.p;
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}


void validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou)
{
	int j;
	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "voc/2007_test.txt");
	char *difficult_valid_images = option_find_str(options, "difficult", NULL);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	char *mapf = option_find_str(options, "map", 0);
	int *map = 0;
	if (mapf) map = read_map(mapf);
	FILE* reinforcement_fd = NULL;

	/*network net = *parse_network_cfg(cfgfile);	// set batch
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	//set_batch_network(&net, 1);
	fuse_conv_batchnorm(net);*/
	
	network *net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);

	fprintf(stderr,"*******************************************1\n");
/*	
    for (j = 0; j < net->n; ++j) {
        layer *l = &net->layers[j];
	fprintf(stderr,"*******************************************1111%g\n",net->n);
        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);
            if (l->batch_normalize) {
                int f;
                for (f = 0; f < l->n; ++f)
                {
                    l->biases[f] = l->biases[f] - (double)l->scales[f] * l->rolling_mean[f] / (sqrt((double)l->rolling_variance[f]) + .000001f);

                    const size_t filter_size = l->size*l->size*l->c;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f*filter_size + i;
                        l->weights[w_index] = (double)l->weights[w_index] * l->scales[f] / (sqrt((double)l->rolling_variance[f]) + .000001f);
                    }
                }

                l->batch_normalize = 0;
#ifdef GPU
                if (gpu_index >= 0) {
                    push_convolutional_layer(*l);
                }
#endif
            }
        }
        else {
            //printf(" Fusion skip layer type: %d \n", l->type);
        }
    }
*/

	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
	srand(time(0));
	
	list *plist = get_paths(valid_images);//get the image path
	char **paths = (char **)list_to_array(plist);

	char **paths_dif = NULL;
	
	//if have difficult valid_image then it will get the difficult_image path
	if (difficult_valid_images) {
		list *plist_dif = get_paths(difficult_valid_images);
		paths_dif = (char **)list_to_array(plist_dif);
	}
	
	layer l = net->layers[net->n-1];
	int classes = l.classes;
	
	int m = plist->size;
	// m is the number of all test image
	int i = 0;
	int t;

	const float thresh = .005;
	const float nms = .45;
	const float iou_thresh = 0.5;

	int nthreads = 4;
	image *val = calloc(nthreads, sizeof(image));//calloc func is to get the memory in
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));
	
	
	load_args args = { 0 };
	args.w = net->w;
	args.h = net->h;
	args.c = net->c;
	args.type = IMAGE_DATA;
	//args.type = LETTERBOX_DATA;

	//const float thresh_calc_avg_iou = 0.24;
	float avg_iou = 0;
	int tp_for_thresh = 0;
	int fp_for_thresh = 0;

	box_prob *detections = calloc(1, sizeof(box_prob));
	int detections_count = 0;
	int unique_truth_count = 0;
	int *truth_classes_count = calloc(classes, sizeof(int));
	
	//load the data with multi-threading
	for (t = 0; t < nthreads; ++t) {
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}
	
	
	double start = what_time_is_it_now();
	for (i = nthreads; i < m + nthreads; i += nthreads) {
		fprintf(stderr, "%d\n", i);
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			pthread_join(thr[t], 0);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}

		for (t = 0; t < nthreads && i + t < m; ++t){
			args.path = paths[i + t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = load_data_in_thread(args);
		}

		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			const int image_index = i + t - nthreads;
			char *path = paths[image_index];
			char *id = basecfg(path);
			float *X = val_resized[t].data;
			network_predict(net, X);

			int nboxes = 0;
			int letterbox = (args.type == LETTERBOX_DATA);
			float hier_thresh = 0;
			int w = val[t].w;
			int h = val[t].h;

			detection *dets = get_network_boxes(net, 1, 1, thresh, hier_thresh, 0, 0, &nboxes);
			//do not have use the letterbox

			//detection *dets = get_network_boxes(&net, val[t].w, val[t].h, thresh, hier_thresh, 0, 1, &nboxes, letterbox); // for letterbox=1
			if (nms) do_nms_sort(dets, nboxes, classes, nms);

			char labelpath[4096];
			replace_image_to_label(path, labelpath);
			int num_labels = 0;
			box_label *truth = read_boxes(labelpath, &num_labels);
			int i, j;
			for (j = 0; j < num_labels; ++j) {
				truth_classes_count[truth[j].id]++;
			}


			// difficult
			box_label *truth_dif = NULL;
			int num_labels_dif = 0;
			if (paths_dif)
			{
				char *path_dif = paths_dif[image_index];
				char labelpath_dif[4096];
				replace_image_to_label(path_dif, labelpath_dif);

				truth_dif = read_boxes(labelpath_dif, &num_labels_dif);
			}
//fprintf(stderr, "66666666\n");
			const int checkpoint_detections_count = detections_count;
			//fprintf(stderr, "66666666\n");
			for (i = 0; i < nboxes; ++i) {
				int class_id;
				for (class_id = 0; class_id < classes; ++class_id) {
					float prob = dets[i].prob[class_id];
					if (prob > 0) {
						detections_count++;
						detections = realloc(detections, detections_count * sizeof(box_prob));
						detections[detections_count - 1].b = dets[i].bbox;
						detections[detections_count - 1].p = prob;
						detections[detections_count - 1].image_index = image_index;
						detections[detections_count - 1].class_id = class_id;
						detections[detections_count - 1].truth_flag = 0;
						detections[detections_count - 1].unique_truth_index = -1;

						int truth_index = -1;
						float max_iou = 0;
						for (j = 0; j < num_labels; ++j)
						{
							box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
							//printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d \n", 
							//	box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
							float current_iou = box_iou(dets[i].bbox, t);
							if (current_iou > iou_thresh && class_id == truth[j].id) {
								if (current_iou > max_iou) {
									max_iou = current_iou;
									truth_index = unique_truth_count + j;
								}
							}
						}

						// best IoU
						if (truth_index > -1) {
							detections[detections_count - 1].truth_flag = 1;
							detections[detections_count - 1].unique_truth_index = truth_index;
						}
						else {
							// if object is difficult then remove detection
							for (j = 0; j < num_labels_dif; ++j) {
								box t = { truth_dif[j].x, truth_dif[j].y, truth_dif[j].w, truth_dif[j].h };
								float current_iou = box_iou(dets[i].bbox, t);
								if (current_iou > iou_thresh && class_id == truth_dif[j].id) {
									--detections_count;
									break;
								}
							}
						}

						// calc avg IoU, true-positives, false-positives for required Threshold
						if (prob > thresh_calc_avg_iou) {
							int z, found = 0;
							for (z = checkpoint_detections_count; z < detections_count-1; ++z)
								if (detections[z].unique_truth_index == truth_index) {
									found = 1; break;
								}

							if(truth_index > -1 && found == 0) {
								avg_iou += max_iou;
								++tp_for_thresh;
							}
							else
								fp_for_thresh++;
						}
					}
				}
			}
				
			unique_truth_count += num_labels;

			//static int previous_errors = 0;
			//int total_errors = fp_for_thresh + (unique_truth_count - tp_for_thresh);
			//int errors_in_this_image = total_errors - previous_errors;
			//previous_errors = total_errors;
			//if(reinforcement_fd == NULL) reinforcement_fd = fopen("reinforcement.txt", "wb");
			//char buff[1000];
			//sprintf(buff, "%s\n", path);
			//if(errors_in_this_image > 0) fwrite(buff, sizeof(char), strlen(buff), reinforcement_fd);

			free_detections(dets, nboxes);
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}
	
       
	if((tp_for_thresh + fp_for_thresh) > 0)
		avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);

	
	// SORT(detections)
	qsort(detections, detections_count, sizeof(box_prob), detections_comparator);
	
	typedef struct {
		double precision;
		double recall;
		int tp, fp, fn;
	} pr_t;
	
	// for PR-curve
	pr_t **pr = calloc(classes, sizeof(pr_t*));
	for (i = 0; i < classes; ++i) {
		pr[i] = calloc(detections_count, sizeof(pr_t));
	}
	printf("detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);


	int *truth_flags = calloc(unique_truth_count, sizeof(int));

	int rank;
	for (rank = 0; rank < detections_count; ++rank) {
		if(rank % 100 == 0)
			printf(" rank = %d of ranks = %d \r", rank, detections_count);

		if (rank > 0) {
			int class_id;
			for (class_id = 0; class_id < classes; ++class_id) {
				pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
				pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
			}
		}

		box_prob d = detections[rank];
		// if (detected && isn't detected before)
		if (d.truth_flag == 1) {
			if (truth_flags[d.unique_truth_index] == 0) 
			{
				truth_flags[d.unique_truth_index] = 1;
				pr[d.class_id][rank].tp++;	// true-positive
			}
		}
		else {
			pr[d.class_id][rank].fp++;	// false-positive
		}

		for (i = 0; i < classes; ++i) 
		{
			const int tp = pr[i][rank].tp;
			const int fp = pr[i][rank].fp;
			const int fn = truth_classes_count[i] - tp;	// false-negative = objects - true-positive
			pr[i][rank].fn = fn;

			if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
			else pr[i][rank].precision = 0;

			if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
			else pr[i][rank].recall = 0;
		}
	}

	free(truth_flags);
	
	
	double mean_average_precision = 0;

	for (i = 0; i < classes; ++i) {
		double avg_precision = 0;
		int point;
		for (point = 0; point < 11; ++point) {
			double cur_recall = point * 0.1;
			double cur_precision = 0;
			for (rank = 0; rank < detections_count; ++rank)
			{
				if (pr[i][rank].recall >= cur_recall) {	// > or >=
					if (pr[i][rank].precision > cur_precision) {
						cur_precision = pr[i][rank].precision;
					}
				}
			}
			//printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_recall, cur_precision);

			avg_precision += cur_precision;
		}
		avg_precision = avg_precision / 11;
		printf("class_id = %d, name = %s, \t ap = %2.2f %% \n", i, names[i], avg_precision*100);
		mean_average_precision += avg_precision;
	}
	
	const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
	const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
	const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
	printf(" for thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
		thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

	printf(" for thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n", 
		thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

	mean_average_precision = mean_average_precision / classes;
	printf("\n mean average precision (mAP) = %f, or %2.2f %% \n", mean_average_precision, mean_average_precision*100);


	for (i = 0; i < classes; ++i) {
		free(pr[i]);
	}
	free(pr);
	free(detections);
	free(truth_classes_count);

	fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
	if (reinforcement_fd != NULL) fclose(reinforcement_fd);
}

#ifdef OPENCV
typedef struct {
	float w, h;
} anchors_t;

int anchors_comparator(const void *pa, const void *pb)
{
	anchors_t a = *(anchors_t *)pa;
	anchors_t b = *(anchors_t *)pb;
	float diff = b.w*b.h - a.w*a.h;
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show)
{
	printf("\n num_of_clusters = %d, width = %d, height = %d \n", num_of_clusters, width, height);
	if (width < 0 || height < 0) {
		printf("Usage: darknet detector calc_anchors data/voc.data -num_of_clusters 9 -width 416 -height 416 \n");
		printf("Error: set width and height \n");
		return;
	}

	//float pointsdata[] = { 1,1, 2,2, 6,6, 5,5, 10,10 };
	float *rel_width_height_array = calloc(1000, sizeof(float));

	list *options = read_data_cfg(datacfg);
	char *train_images = option_find_str(options, "train", "data/train.list");
	list *plist = get_paths(train_images);
	int number_of_images = plist->size;
	char **paths = (char **)list_to_array(plist);

	int number_of_boxes = 0;
	printf(" read labels from %d images \n", number_of_images);

	int i, j;
	for (i = 0; i < number_of_images; ++i) {
		char *path = paths[i];
		char labelpath[4096];
		find_replace(path, "images", "labels", labelpath);
        	find_replace(labelpath, "JPEGImages", "labels", labelpath);
	        find_replace(labelpath, ".jpg", ".txt", labelpath);
        	find_replace(labelpath, ".JPEG", ".txt", labelpath);

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		//printf(" new path: %s \n", labelpath);
		char buff[1024];
		for (j = 0; j < num_labels; ++j)
		{
			if (truth[j].x > 1 || truth[j].x <= 0 || truth[j].y > 1 || truth[j].y <= 0 ||
				truth[j].w > 1 || truth[j].w <= 0 || truth[j].h > 1 || truth[j].h <= 0) 
			{				
				printf("\n\nWrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f \n",
					labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
				sprintf(buff, "echo \"Wrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f\" >> bad_label.list", 
					labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
				system(buff);				
			}
			number_of_boxes++;
			rel_width_height_array = realloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));
			rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
			rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
			printf("\r loaded \t image: %d \t box: %d", i+1, number_of_boxes);
		}
	}
	printf("\n all loaded. \n");

	CvMat* points = cvCreateMat(number_of_boxes, 2, CV_32FC1);
	CvMat* centers = cvCreateMat(num_of_clusters, 2, CV_32FC1);
	CvMat* labels = cvCreateMat(number_of_boxes, 1, CV_32SC1);

	for (i = 0; i < number_of_boxes; ++i) {
		points->data.fl[i * 2] = rel_width_height_array[i * 2];
		points->data.fl[i * 2 + 1] = rel_width_height_array[i * 2 + 1];
		//cvSet1D(points, i * 2, cvScalar(rel_width_height_array[i * 2], 0, 0, 0));
		//cvSet1D(points, i * 2 + 1, cvScalar(rel_width_height_array[i * 2 + 1], 0, 0, 0));
	}


	const int attemps = 10;
	double compactness;

	enum {
		KMEANS_RANDOM_CENTERS = 0,
		KMEANS_USE_INITIAL_LABELS = 1,
		KMEANS_PP_CENTERS = 2
	};
	
	printf("\n calculating k-means++ ...");
	// Should be used: distance(box, centroid) = 1 - IoU(box, centroid)
	cvKMeans2(points, num_of_clusters, labels, 
		cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10000, 0), attemps, 
		0, KMEANS_PP_CENTERS,
		centers, &compactness);

	// sort anchors
	qsort(centers->data.fl, num_of_clusters, 2*sizeof(float), anchors_comparator);

	//orig 2.0 anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
	//float orig_anch[] = { 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52 };
	// worse than ours (even for 19x19 final size - for input size 608x608)

	//orig anchors = 1.3221,1.73145, 3.19275,4.00944, 5.05587,8.09892, 9.47112,4.84053, 11.2364,10.0071
	//float orig_anch[] = { 1.3221,1.73145, 3.19275,4.00944, 5.05587,8.09892, 9.47112,4.84053, 11.2364,10.0071 };
	// orig (IoU=59.90%) better than ours (59.75%)

	//gen_anchors.py = 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66
	//float orig_anch[] = { 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66 };

	// ours: anchors = 9.3813,6.0095, 3.3999,5.3505, 10.9476,11.1992, 5.0161,9.8314, 1.5003,2.1595
	//float orig_anch[] = { 9.3813,6.0095, 3.3999,5.3505, 10.9476,11.1992, 5.0161,9.8314, 1.5003,2.1595 };
	//for (i = 0; i < num_of_clusters * 2; ++i) centers->data.fl[i] = orig_anch[i];
	
	//for (i = 0; i < number_of_boxes; ++i)
	//	printf("%2.2f,%2.2f, ", points->data.fl[i * 2], points->data.fl[i * 2 + 1]);

	printf("\n");
	float avg_iou = 0;
	for (i = 0; i < number_of_boxes; ++i) {
		float box_w = points->data.fl[i * 2];
		float box_h = points->data.fl[i * 2 + 1];
		//int cluster_idx = labels->data.i[i];		
		int cluster_idx = 0;
		float min_dist = FLT_MAX;
		for (j = 0; j < num_of_clusters; ++j) {
			float anchor_w = centers->data.fl[j * 2];
			float anchor_h = centers->data.fl[j * 2 + 1];
			float w_diff = anchor_w - box_w;
			float h_diff = anchor_h - box_h;
			float distance = sqrt(w_diff*w_diff + h_diff*h_diff);
			if (distance < min_dist) min_dist = distance, cluster_idx = j;
		}
		
		float anchor_w = centers->data.fl[cluster_idx * 2];
		float anchor_h = centers->data.fl[cluster_idx * 2 + 1];
		float min_w = (box_w < anchor_w) ? box_w : anchor_w;
		float min_h = (box_h < anchor_h) ? box_h : anchor_h;
		float box_intersect = min_w*min_h;
		float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
		float iou = box_intersect / box_union;
		if (iou > 1 || iou < 0) { // || box_w > width || box_h > height) {
			printf(" Wrong label: i = %d, box_w = %d, box_h = %d, anchor_w = %d, anchor_h = %d, iou = %f \n",
				i, box_w, box_h, anchor_w, anchor_h, iou);
		}
		else avg_iou += iou;
	}
	avg_iou = 100 * avg_iou / number_of_boxes;
	printf("\n avg IoU = %2.2f %% \n", avg_iou);

	char buff[1024];
	FILE* fw = fopen("anchors.txt", "wb");
	printf("\nSaving anchors to the file: anchors.txt \n");
	printf("anchors = ");
	for (i = 0; i < num_of_clusters; ++i) {
		sprintf(buff, "%2.4f,%2.4f", centers->data.fl[i * 2], centers->data.fl[i * 2 + 1]);
		printf("%s", buff);
		fwrite(buff, sizeof(char), strlen(buff), fw);
		if (i + 1 < num_of_clusters) {
			fwrite(", ", sizeof(char), 2, fw);
			printf(", ");
		}
	}
	printf("\n");
	fclose(fw);

	if (show) {
		size_t img_size = 700;
		IplImage* img = cvCreateImage(cvSize(img_size, img_size), 8, 3);
		cvZero(img);
		for (j = 0; j < num_of_clusters; ++j) {
			CvPoint pt1, pt2;
			pt1.x = pt1.y = 0;
			pt2.x = centers->data.fl[j * 2] * img_size / width;
			pt2.y = centers->data.fl[j * 2 + 1] * img_size / height;
			cvRectangle(img, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
		}

		for (i = 0; i < number_of_boxes; ++i) {
			CvPoint pt;
			pt.x = points->data.fl[i * 2] * img_size / width;
			pt.y = points->data.fl[i * 2 + 1] * img_size / height;
			int cluster_idx = labels->data.i[i];
			int red_id = (cluster_idx * (uint64_t)123 + 55) % 255;
			int green_id = (cluster_idx * (uint64_t)321 + 33) % 255;
			int blue_id = (cluster_idx * (uint64_t)11 + 99) % 255;
			cvCircle(img, pt, 1, CV_RGB(red_id, green_id, blue_id), CV_FILLED, 8, 0);
			//if(pt.x > img_size || pt.y > img_size) printf("\n pt.x = %d, pt.y = %d \n", pt.x, pt.y);
		}
		cvShowImage("clusters", img);
		cvWaitKey(0);
		cvReleaseImage(&img);
		cvDestroyAllWindows();
	}

	free(rel_width_height_array);
	cvReleaseMat(&points);
	cvReleaseMat(&centers);
	cvReleaseMat(&labels);
}
#else
void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show) {
	printf(" k-means++ can't be used without OpenCV, because there is used cvKMeans2 implementation \n");
}
#endif // OPENCV


void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "2007_test.txt");
    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < l.w*l.h*l.n; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            if (best_iou <= 100 && best_iou == best_iou) avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
	fprintf(stderr, "proposals:%5d\tPrecision:%.2f%%\n",proposals,100.*correct/(float)proposals);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}


void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];
        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL);
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions");
            cvWaitKey(0);
            cvDestroyAllWindows();
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

/*
void censor_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL);
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = in_s.data;
        network_predict(net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 0, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int left  = b.x-b.w/2.;
                int top   = b.y-b.h/2.;
                censor_image(in, left, top, b.w, b.h);
            }
        }
        show_image(in, base);
        cvWaitKey(10);
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}

void extract_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL);
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    int count = 0;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        show_image(in, base);

        int nboxes = 0;
        float *X = in_s.data;
        network_predict(net, X);
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 1, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int size = b.w*in.w > b.h*in.h ? b.w*in.w : b.h*in.h;
                int dx  = b.x*in.w-size/2.;
                int dy  = b.y*in.h-size/2.;
                image bim = crop_image(in, dx, dy, size, size);
                char buff[2048];
                sprintf(buff, "results/extract/%07d", count);
                ++count;
                save_image(bim, buff);
                free_image(bim);
            }
        }
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}
*/

/*
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets)
{
    network_predict_image(net, im);
    layer l = net->layers[net->n-1];
    int nboxes = num_boxes(net);
    fill_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 0, dets);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
}
*/

void run_detector(int argc, char **argv)
{
    int dont_show = find_arg(argc, argv, "-dont_show");
    int show = find_arg(argc, argv, "-show");
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear"); //so what do the parameter mean,does it mean something ,i do not understand ,and what is the default set
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights);
    else if(0==strcmp(argv[2], "map")) validate_detector_map(datacfg, cfg, weights,thresh);
    else if(0==strcmp(argv[2], "calc_anchors"))  calc_anchors(datacfg, num_of_clusters, width, height, show);//there is warning ,so may be something is wrong
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    //else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}

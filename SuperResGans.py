import tensorflow as tf
import glob
import cv2 as cv
import os
import numpy as np
import time
import datetime
import matplotlib.pytplot as plt

paths = glob.glob('D:/Arjun/Python/flickr-image-dataset/flickr30k_images/flickr30k_images/*.jpg')

batch_size = 20
n = 10000   #no of images to be takes from the dataset
h = 256
w = 256

# def zero_pad(img,blur):
#     """
#     args - img,blur
#     zero pads img and blur to h,w size
#     return the args.
#     """
#     img_shape = img.shape
    
#     rpad = np.abs(img_shape[0]-h)
#     cpad = np.abs(img_shape[1]-w)
    
#     img = np.pad(img, ((np.ceil(rpad/2).astype(np.int32), rpad//2), (np.ceil(cpad/2).astype(np.int32), cpad//2),(0,0)), 'constant',constant_values= 0)
#     blur = np.pad(blur, ((np.ceil(rpad/2).astype(np.int32), rpad//2), (np.ceil(cpad/2).astype(np.int32), cpad//2),(0,0)), 'constant',constant_values= 0)

#     return img,blur



def load_data(path):
    """
    args- path: of each image
    loading each image, blurring.
    returns original and the blurred version of the image
    """
    img = cv.imread(path)
    blur = cv.GaussianBlur(img, (5,5),0)
    #img, blur = zero_pad(img,blur)  --> with zero pad
    img = cv.resize(img, (256,256))
    blur = cv.resize(blur, (256,256))   #resizing

    return img,blur

def create_batch(batch_num):
    """
    args- batch_num
    creates (batch_num)th batch of the defined batch size
    returns a batch of original and blurred images in format -[batch_size,h,w,channels] 
    """
    batch_real = []
    batch_blur = []

    try:
        start = batch_size*(batch_num-1)
        end = batch_size*batch_num
        bpath = paths[start:end]
    except:
        bpath = paths[start:]                #index out of bound
    
    for path in bpath:
        img,blur = load_data(path)
        batch_real.append(img)
        batch_blur.append(blur)
    
    return batch_real, batch_blur

# def spp(inp, bins):
#     """
#     Spatial pyramidal pooling (kaiming 2015).
#     unable to implement - https://github.com/tensorflow/tensorflow/issues/1967
#     ksize has to be constant!!
    
#     """
#     shape = tf.shape(inp)
#     with tf.name_scope("spp"):

#         spp_1 = tf.nn.max_pool(inp, [1,tf.cast(tf.ceil(shape[1]/bins[0]),dtype = tf.int64), (tf.ceil(shape[2]/bins[0])),1], [1, shape[1]//bins[0], shape[2]//bins[0], 1], padding = 'SAME')
#         spp_2 = tf.nn.max_pool(inp, [1,tf.cast(tf.ceil(shape[1]/bins[1]),dtype = tf.int64), (tf.ceil(shape[2]/bins[1])),1], [1, shape[1]//bins[1], shape[2]//bins[1], 1], padding = 'SAME')
#         spp_3 = tf.nn.max_pool(inp, [1,tf.cast(tf.ceil(shape[1]/bins[2]),dtype = tf.int64), (tf.ceil(shape[2]/bins[2])),1], [1, shape[1]//bins[2], shape[2]//bins[2], 1], padding = 'SAME')
#         spp_4 = tf.nn.max_pool(inp, [1,tf.cast(tf.ceil(shape[1]/bins[3]),dtype = tf.int64), (tf.ceil(shape[2]/bins[3])),1], [1, shape[1]//bins[3], shape[2]//bins[3], 1], padding = 'SAME')

#         spp_1_flat = tf.reshape(spp_1, [shape[0], -1])
#         spp_2_flat = tf.reshape(spp_2, [shape[0], -1])
#         spp_3_flat = tf.reshape(spp_3, [shape[0], -1])
#         spp_4_flat = tf.reshape(spp_4, [shape[0], -1])

#         spp_pool = tf.concat(values = [spp_1_flat,spp_2_flat,spp_3_flat,spp_4_flat], axis = 1)

#         return spp_pool

def batch_norm(x,gamma,beta,is_training):

    decay = 0.99

    with tf.name_scope("batch_norm"):
        pop_mean = tf.Variable(tf.zeros(tf.shape(gamma)), trainable = False)
        pop_var = tf.Variable(tf.ones(tf.shape(gamma)), trainable = False)

        mean, var = tf.nn.moments(x, [0])
        if is_training:
            
            train_mean = tf.assign(pop_mean, pop_mean*decay + (1-decay)*mean)
            train_var = tf.assign(pop_var, pop_var*decay + (1-decay)*var)
            
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(x, mean, var, beta, gamma, 0.01)
        else:
            return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, gamma, 0.01)


real = tf.placeholder(shape = [None,h,w,3], dtype=tf.float32, name = 'real')
blur = tf.placeholder(shape = [None,h,w,3], dtype=tf.float32, name = 'blur')

def generator(x):
    """
    args- x:[batch_size,h,w,channels] ------> image
    generator network model
    all layer prefixed with g
    returns - the generated output and trainables (generated output is same shape as input)
    """
    with tf.name_scope("gconv1"):
        wc_1 = tf.Variable(tf.random_uniform([9,9,3,64],-2,2), name = 'wc_1')
        conv_1 = tf.nn.conv2d(x, wc_1, [1,1,1,1], padding = "SAME")
        act_1 = tf.nn.leaky_relu(conv_1)

    with tf.name_scope("g_res_1"):
        
        wc_2 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_2')
        conv_2 = tf.nn.conv2d(act_1, wc_2, [1,1,1,1], padding = 'SAME')
        
        gamma_1 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_1')
        beta_1 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_1')
        bn_1 = batch_norm(conv_2, gamma_1, beta_1, True)
        
        act_2 = tf.nn.leaky_relu(bn_1)
        
        
        wc_3 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_3')
        conv_3 = tf.nn.conv2d(act_2, wc_3, [1,1,1,1], padding = 'SAME')
        
        gamma_2 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_2')
        beta_2 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_2')
        bn_2 = batch_norm(conv_3, gamma_2, beta_2, True)

        res_1 = bn_2 + act_1                            #residual mapping

        

    with tf.name_scope("g_res_2"):
        
        wc_4 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_4')
        conv_4 = tf.nn.conv2d(res_1, wc_4, [1,1,1,1], padding = 'SAME')
        
        gamma_3 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_3')
        beta_3 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_3')
        bn_3 = batch_norm(conv_4, gamma_3, beta_3, True)
        
        act_3 = tf.nn.leaky_relu(bn_3)
        
        
        wc_5 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_5')
        conv_5 = tf.nn.conv2d(act_3, wc_5, [1,1,1,1], padding = 'SAME')
        
        gamma_4 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_4')
        beta_4 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_4')
        bn_4 = batch_norm(conv_5, gamma_4, beta_4, True)

        res_2 = bn_4 + res_1                                        #residual mapping
    
    with tf.name_scope("g_res_3"):
        
        wc_6 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_6')
        conv_6 = tf.nn.conv2d(res_2, wc_6, [1,1,1,1], padding = 'SAME')
        
        gamma_5 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_5')
        beta_5 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_5')
        bn_5 = batch_norm(conv_6, gamma_5, beta_5, True)
        
        act_4 = tf.nn.leaky_relu(bn_5)
        
        
        wc_7 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_7')
        conv_7 = tf.nn.conv2d(act_4, wc_7, [1,1,1,1], padding = 'SAME')
        
        gamma_6 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_6')
        beta_6 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_6')
        bn_6 = batch_norm(conv_7, gamma_6, beta_6, True)

        res_3 = bn_6 + res_2                                    #residual mapping
    

    with tf.name_scope("g_res_4"):
        
        wc_8 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_8')
        conv_8 = tf.nn.conv2d(res_3, wc_8, [1,1,1,1], padding = 'SAME')
        
        gamma_7 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_7')
        beta_7 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_7')
        bn_7 = batch_norm(conv_8, gamma_7, beta_7, True)
        
        act_5 = tf.nn.leaky_relu(bn_7)
        
        
        wc_9 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_9')
        conv_9 = tf.nn.conv2d(act_5, wc_9, [1,1,1,1], padding = 'SAME')
        
        gamma_8 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_8')
        beta_8 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_8')
        bn_8 = batch_norm(conv_9, gamma_8, beta_8, True)

        res_4 = bn_8 + res_3                                #residual mapping
    
    with tf.name_scope("g_res_5"):
        
        wc_10 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_10')
        conv_10 = tf.nn.conv2d(res_4, wc_10, [1,1,1,1], padding = 'SAME')
        
        gamma_9 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_9')
        beta_9 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_9')
        bn_9 = batch_norm(conv_10, gamma_9, beta_9, True)
        
        act_6 = tf.nn.leaky_relu(bn_9)
        
        
        wc_11 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_11')
        conv_11 = tf.nn.conv2d(act_6, wc_11, [1,1,1,1], padding = 'SAME')
        
        gamma_10 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_10')
        beta_10 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_10')
        bn_10 = batch_norm(conv_11, gamma_10, beta_10, True)

        res_5 = bn_10 + res_4                               #residual mapping
    
    with tf.name_scope("g_convbn_skip"):
        
        wc_12 = tf.Variable(tf.random_uniform([3,3,64,64],-2,2), name = 'wc_12')
        conv_12 = tf.nn.conv2d(res_5, wc_12, [1,1,1,1], padding = 'SAME')
        
        gamma_11 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'gamma_11')
        beta_11 = tf.Variable(tf.random_uniform([256,256,64],-2,2), name = 'beta_11')
        bn_11 = batch_norm(conv_12, gamma_11, beta_11, True)

        res_6 = bn_11 + act_1                               #residual mapping

    with tf.name_scope("goutconv-3channel"):        
        wc_o = tf.Variable(tf.random_uniform([9,9,64,3],-2,2), name = 'wc_o')
        out = tf.nn.conv2d(res_6,wc_o, [1,1,1,1], padding = 'SAME')
        cout = tf.nn.leaky_relu(out, name = "output")

    return cout, [wc_1, wc_2, wc_3, wc_4, wc_5, wc_6, wc_7, wc_8, wc_9, wc_10, wc_11, wc_12, wc_o, gamma_1, gamma_2,
                    gamma_3, gamma_4, gamma_5, gamma_6, gamma_7, gamma_8, gamma_9, gamma_10, gamma_11, beta_1, beta_2,
                    beta_3, beta_4, beta_5, beta_6, beta_7, beta_8, beta_9, beta_10, beta_11]



def discriminator(x):
    """
    args- x:[batch_size,h,w,channels] ------> image
    discriminator network model
    all layer prefixed with d
    returns - the classification output and trainables
    """

    with tf.name_scope("dconv1"):
        wc_1 = tf.Variable(tf.random_normal([3,3,3,32]), name = 'wc_1')
        conv_1 = tf.nn.conv2d(x, wc_1, [1,2,2,1], padding = 'SAME')
        act_1 = tf.nn.leaky_relu(conv_1)
        mpool_1 = tf.nn.max_pool(act_1, [1,2,2,1], [1,2,2,1], padding = 'SAME')

    with tf.name_scope("dconv2"):
        wc_2 = tf.Variable(tf.random_normal([3,3,32,64]), name = 'wc_2')
        conv_2 = tf.nn.conv2d(mpool_1, wc_2, [1,2,2,1], padding = 'SAME')
        act_2 = tf.nn.leaky_relu(conv_2)
        mpool_2 = tf.nn.max_pool(act_2, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    
    with tf.name_scope("dconv3"):
        wc_3 = tf.Variable(tf.random_normal([3,3,64,128]), name = 'wc_3')
        conv_3 = tf.nn.conv2d(mpool_2, wc_3, [1,2,2,1], padding = 'SAME')
        act_3 = tf.nn.leaky_relu(conv_3)
        mpool_3 = tf.nn.max_pool(act_3, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    
    with tf.name_scope("dconv4"):
        wc_4 = tf.Variable(tf.random_normal([3,3,128,256]), name = 'wc_4')
        conv_4 = tf.nn.conv2d(mpool_3, wc_4, [1,2,2,1], padding = 'SAME')
        act_4 = tf.nn.leaky_relu(conv_4)
        mpool_4 = tf.nn.max_pool(act_4, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    
    # with tf.name_scope("dconv5"):    --->resource exhaustion error
    #     wc_5 = tf.Variable(tf.random_normal([3,3,256,512]), name = 'wc_5')
    #     conv_5 = tf.nn.conv2d(mpool_4, wc_5, [1,2,2,1], padding = 'SAME')
    #     act_5 = tf.nn.leaky_relu(conv_5)
    #     mpool_5 = tf.nn.max_pool(act_5, [1,2,2,1], [1,2,2,1], padding = 'SAME')

    flatten = tf.reshape(mpool_4,[batch_size,-1])
    # spp_pool = spp(act_4,[6,4,2,1])  --> with spp


    with tf.name_scope("ddense1"):
        w_1 = tf.Variable(tf.random_normal([256,128]), name = 'w_1')
        #w_1 = tf.Variable(tf.random_normal([57, 128]), name = 'w_1')  -->with spp
        b_1 = tf.Variable(tf.random_normal([128]), name = 'b_1')
        tf.summary.histogram("w_1",w_1)
        tf.summary.histogram("b_1",b_1)

        layer_1 = tf.nn.leaky_relu((tf.matmul(flatten, w_1)+b_1))
        #layer_1 = tf.nn.leaky_relu((tf.matmul(spp_pool,w_1)+b_1))  --> with spp

    with tf.name_scope("ddense2"):
        w_2 = tf.Variable(tf.random_normal([128,64]), name = 'w_2')
        b_2 = tf.Variable(tf.random_normal([64]), name = 'b_2')
        tf.summary.histogram("w_2",w_2)
        tf.summary.histogram("b_2",b_2)

        layer_2 = tf.nn.leaky_relu((tf.matmul(layer_1,w_2)+b_2))

    with tf.name_scope("doutput"):
        w_3 = tf.Variable(tf.random_normal([64,1]), name = 'w_3')
        b_3 = tf.Variable(tf.random_normal([1]), name = 'b_3')
        tf.summary.histogram("w_3",w_3)
        tf.summary.histogram("b_3",b_3)

        out = tf.nn.leaky_relu((tf.matmul(layer_2,w_3)+b_3))

    return out, [wc_1, wc_2, wc_3, wc_4, w_1, w_2, w_3, b_1, b_2, b_3]

def train():
    """
    function that trains the generator and discriminator
    loss function based on (Ledig 2017)
    optimzer - adam
    """
    gz, gvl = generator(blur)
    r_out, dvl = discriminator(real)
    f_out, dvl = discriminator(gz)
    
    with tf.name_scope("cost"):
        fake_dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(f_out),logits = f_out))
        real_dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(r_out),logits = r_out))
        
        tf.summary.scalar("fake_dloss",fake_dloss)
        tf.summary.scalar("real_dloss",real_dloss)
        
        dloss = fake_dloss + real_dloss
        
        gloss = tf.reduce_mean(tf.math.square(real-gz)) + tf.math.pow(10.0,-3)*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(f_out),logits = f_out))
        tf. summary.scalar("gloss",gloss)
    
    with tf.name_scope("optimizer"):
        dis_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, name = 'doptimizer')
        gen_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, name = 'goptimizer')
        
        dgrads = dis_optimizer.compute_gradients(dloss, var_list = dvl)
        ggrads = gen_optimizer.compute_gradients(gloss, var_list = gvl)
        
        for g in dgrads:
            tf.summary.histogram("{} grad".format(g[1].name),g[0])
        for g in ggrads:                                                        #plotting the gradients
            tf.summary.histogram("{} grad".format(g[1].name), g[0])

        dis_opt = dis_optimizer.apply_gradients(dgrads)
        gen_opt = gen_optimizer.apply_gradients(ggrads)

    
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables(),max_to_keep = 5, keep_checkpoint_every_n_hours = 1)

    nepochs = 30
    

    with tf.Session() as sess:
        
        
        sess.run(tf.global_variables_initializer())
        
        try:
          saver = tf.train.import_meta_graph('/tmp/model.ckpt.meta')     #restoring model
          saver.restore(sess,'/tmp/model.ckpt')
        except:
          print("no saved model")

        writer = tf.summary.FileWriter('logs',graph = sess.graph)
        
        for _ in range(nepochs):
            
            i = 1
            prev_sum = 0

            while i<(n//batch_size):
                
                start = time.time()
                
                print("batch: ",i)

                batch_real, batch_blur = create_batch(i)
                
                _,dc = sess.run([dis_opt,dloss], feed_dict = {blur: np.array(batch_blur), real: np.array(batch_real)})
                _,gc,summary,gout = sess.run([gen_opt,gloss,merged, gz], feed_dict = {blur:np.array(batch_blur), real: np.array(batch_real)})
                

                writer.add_summary(summary,i)
                saver.save(sess,'/tmp/model.ckpt',global_step = i)

                if i%10==0:
                  j =0
                
                  for x,y in zip(batch_blur, batch_real):
                    fig,axs = plt.subplots(1,3, figsize = (15,15))
                    axs[0].imshow(cv.cvtColor(x, cv.COLOR_BGR2RGB))
                    axs[1].imshow(cv.cvtColor(gout[j,:,:,:], cv.COLOR_BGR2RGB))
                    axs[2].imshow(cv.cvtColor(y, cv.COLOR_BGR2RGB))
                    j+=1
                    plt.show()
                
                end = time.time()
                prev_sum+=(end-start)
                print('\n')
                print("Eta: ",str(datetime.timedelta(seconds =(prev_sum/i)*((n//batch_size)-i))))
                
                i+=1
                   
                print("discriminator cost: ",dc)
                print("generator cost: ",gc)
                print('\n')
        writer.close()



train()        
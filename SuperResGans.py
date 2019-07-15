import tensorflow as tf
import glob
import cv2 as cv
import os
import numpy as np

paths = glob.glob('D:/Arjun/Python/flickr-image-dataset/flickr30k_images/flickr30k_images/*.jpg')

batch_size = 10
n = 500   #no of images to be takes from the dataset
h = 512
w = 512

def zero_pad(img,blur):
    """
    args - img,blur
    zero pads img and blur to h,w size
    return the args.
    """
    img_shape = img.shape
    
    rpad = np.abs(img_shape[0]-h)
    cpad = np.abs(img_shape[1]-w)
    
    img = np.pad(img, ((np.ceil(rpad/2).astype(np.int32), rpad//2), (np.ceil(cpad/2).astype(np.int32), cpad//2),(0,0)), 'constant',constant_values= 0)
    blur = np.pad(blur, ((np.ceil(rpad/2).astype(np.int32), rpad//2), (np.ceil(cpad/2).astype(np.int32), cpad//2),(0,0)), 'constant',constant_values= 0)

    return img,blur



def load_data(path):
    """
    args- path: of each image
    loading each image, blurring.
    returns original and the blurred version of the image
    """
    img = cv.imread(path)
    blur = cv.GaussianBlur(img, (5,5),0)
    img, blur = zero_pad(img,blur)
    
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




real = tf.placeholder(shape = [None,h,w,3],dtype=tf.float32, name = 'real')
blur = tf.placeholder(shape = [None,h,w,3],dtype=tf.float32,name = 'blur')

def generator(x):
    """
    args- x:[batch_size,h,w,channels] ------> image
    generator network model
    all layer prefixed with g
    returns - the generated output output and trainables (generated output is same shape as input)

    """
    with tf.name_scope("gconv1"):
        wc_1 = tf.Variable(tf.random_normal([3,3,3,32]))
        conv_1 = tf.nn.conv2d(x, wc_1, [1,1,1,1], padding = "SAME")
        act_1 = tf.nn.relu(conv_1)
        mpool_1 = tf.nn.max_pool(act_1, [1,2,2,1], [1,2,2,1], padding = 'SAME')

    with tf.name_scope("gconv2"):
        wc_2 = tf.Variable(tf.random_normal([3,3,32,64]))
        conv_2 = tf.nn.conv2d(mpool_1, wc_2, [1,1,1,1], padding = 'SAME')
        act_2 = tf.nn.relu(conv_2)
        mpool_2 = tf.nn.max_pool(act_2, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    
    with tf.name_scope("gconv3"):
        wc_3 = tf.Variable(tf.random_normal([3,3,64,128]))
        conv_3 = tf.nn.conv2d(mpool_2, wc_3, [1,1,1,1], padding = 'SAME')
        act_3 = tf.nn.relu(conv_3)
        mpool_3 = tf.nn.max_pool(act_3, [1,2,2,1], [1,2,2,1], padding = 'SAME')

    with tf.name_scope("gconv4"):    
        wc_4 = tf.Variable(tf.random_normal([3,3,128,256]))
        conv_4 = tf.nn.conv2d(mpool_3, wc_4, [1,1,1,1], padding = 'SAME')
        act_4 = tf.nn.relu(conv_4)
        mpool_4 = tf.nn.max_pool(act_4, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    
    with tf.name_scope("gconvnopool"):
        wc_5 = tf.Variable(tf.random_normal([3,3,256,256]))
        conv_5 = tf.nn.conv2d(mpool_4, wc_5, [1,1,1,1], padding = 'SAME')
        act_5 = tf.nn.relu(conv_5)

    with tf.name_scope("gupconv1"):
        us_1 = tf.image.resize_nearest_neighbor(act_5, (2*act_5.shape.as_list()[1],2*act_5.shape.as_list()[2]))     #tf backend implentation of keras upsmaple2d
        wc_6 = tf.Variable(tf.random_normal([3,3,256,256]))
        conv_6 = tf.nn.conv2d(us_1, wc_6, [1,1,1,1], padding = 'SAME')
        act_6 = tf.nn.relu(conv_6)
    
    with tf.name_scope("gupconv2"):
        us_2 = tf.image.resize_nearest_neighbor(act_6, (2*act_6.shape.as_list()[1],2*act_6.shape.as_list()[2]))
        wc_7 = tf.Variable(tf.random_normal([3,3,256,128]))
        conv_7 = tf.nn.conv2d(us_2, wc_7, [1,1,1,1], padding = 'SAME')
        act_7 = tf.nn.relu(conv_7)
    
    with tf.name_scope("gupconv3"):
        us_3 = tf.image.resize_nearest_neighbor(act_7, (2*act_7.shape.as_list()[1],2*act_7.shape.as_list()[2]))
        wc_8 = tf.Variable(tf.random_normal([3,3,128,64]))
        conv_8 = tf.nn.conv2d(us_3, wc_8, [1,1,1,1], padding = 'SAME')
        act_8 = tf.nn.relu(conv_8)

    with tf.name_scope("gupconv4"):
        us_4 = tf.image.resize_nearest_neighbor(act_8, (2*act_8.shape.as_list()[1],2*act_8.shape.as_list()[2]))
        wc_9 = tf.Variable(tf.random_normal([3,3,64,32]))
        conv_9 = tf.nn.conv2d(us_4, wc_9, [1,1,1,1], padding = 'SAME')
        act_9 = tf.nn.relu(conv_9)

    with tf.name_scope("goutconv-3channel"):        
        wc_o = tf.Variable(tf.random_normal([3,3,32,3]))
        out = tf.nn.conv2d(act_9,wc_o, [1,1,1,1], padding = 'SAME')

    return out,[wc_1, wc_2, wc_3, wc_4, wc_5, wc_6, wc_7, wc_8, wc_9]



def discriminator(x):
    """
    args- x:[batch_size,h,w,channels] ------> image
    discriminator network model
    all layer prefixed with d
    returns - the classification output and trainables
    """

    with tf.name_scope("dconv1"):
        wc_1 = tf.Variable(tf.random_normal([3,3,3,32]))
        conv_1 = tf.nn.conv2d(x, wc_1, [1,2,2,1], padding = 'SAME')
        act_1 = tf.nn.relu(conv_1)
        mpool_1 = tf.nn.max_pool(act_1, [1,2,2,1], [1,2,2,1], padding = 'SAME')

    with tf.name_scope("dconv2"):
        wc_2 = tf.Variable(tf.random_normal([3,3,32,64]))
        conv_2 = tf.nn.conv2d(mpool_1, wc_2, [1,2,2,1], padding = 'SAME')
        act_2 = tf.nn.relu(conv_2)
        mpool_2 = tf.nn.max_pool(act_2, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    
    with tf.name_scope("dconv3"):
        wc_3 = tf.Variable(tf.random_normal([3,3,64,128]))
        conv_3 = tf.nn.conv2d(mpool_2, wc_3, [1,2,2,1], padding = 'SAME')
        act_3 = tf.nn.relu(conv_3)
        mpool_3 = tf.nn.max_pool(act_3, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    
    with tf.name_scope("dconv4"):
        wc_4 = tf.Variable(tf.random_normal([3,3,128,256]))
        conv_4 = tf.nn.conv2d(mpool_3, wc_4, [1,2,2,1], padding = 'SAME')
        act_4 = tf.nn.relu(conv_4)
        mpool_4 = tf.nn.max_pool(act_4, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    
    # with tf.name_scope("dconv5"):    --->resource exhaustion error
    #     wc_5 = tf.Variable(tf.random_normal([3,3,256,512]))
    #     conv_5 = tf.nn.conv2d(mpool_4, wc_5, [1,2,2,1], padding = 'SAME')
    #     act_5 = tf.nn.relu(conv_5)
    #     mpool_5 = tf.nn.max_pool(act_5, [1,2,2,1], [1,2,2,1], padding = 'SAME')

    flatten = tf.reshape(mpool_4,[batch_size,-1])
    # spp_pool = spp(act_4,[6,4,2,1])  --> with spp


    with tf.name_scope("ddense1"):
        w_1 = tf.Variable(tf.random_normal([1024,128]))
        #w_1 = tf.Variable(tf.random_normal([57, 128]))  -->with spp
        b_1 = tf.Variable(tf.random_normal([128]))
        tf.summary.histogram("w_1",w_1)
        tf.summary.histogram("b_1",b_1)

        layer_1 = tf.nn.relu((tf.matmul(flatten, w_1)+b_1))
        #layer_1 = tf.nn.relu((tf.matmul(spp_pool,w_1)+b_1))  --> with spp

    with tf.name_scope("ddense2"):
        w_2 = tf.Variable(tf.random_normal([128,64]))
        b_2 = tf.Variable(tf.random_normal([64]))
        tf.summary.histogram("w_2",w_2)
        tf.summary.histogram("b_2",b_2)

        layer_2 = tf.nn.relu((tf.matmul(layer_1,w_2)+b_2))

    with tf.name_scope("doutput"):
        w_3 = tf.Variable(tf.random_normal([64,1]))
        b_3 = tf.Variable(tf.random_normal([1]))
        tf.summary.histogram("w_3",w_3)
        tf.summary.histogram("b_3",b_3)

        out = tf.nn.relu((tf.matmul(layer_2,w_3)+b_3))

    return out,[wc_1, wc_2, wc_3, wc_4, w_1, w_2, w_3, b_1, b_2, b_3]

def train():
    """
    function that trains the generator and discriminator
    loss function based on (Goodfellow 2014)
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
        
        gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(f_out),logits = f_out))
        tf. summary.scalar("gloss",gloss)
    
    with tf.name_scope("optimizer"):
        dis_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001, name = 'doptimizer')
        gen_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001, name = 'goptimizer')
        
        dgrads = dis_optimizer.compute_gradients(dloss, var_list = dvl)
        ggrads = gen_optimizer.compute_gradients(gloss, var_list = gvl)
        
        for g in dgrads:
            tf.summary.histogram("{} grad".format(g[1].name),g[0])
        for g in ggrads:                                                        #plotting the gradients
            tf.summary.histogram("{} grad".format(g[1].name), g[0])

        dis_opt = dis_optimizer.apply_gradients(dgrads)
        gen_opt = gen_optimizer.apply_gradients(ggrads)

    
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables(),max_to_keep = 3, keep_checkpoint_every_n_hours = 1)

    nepochs = 1
    

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        writer = tf.summary.FileWriter('logs',graph = sess.graph)
        
        for _ in range(nepochs):
            
            i = 1
            
            while i<(n//batch_size):
                print("batch: ",i)

                batch_real, batch_blur = create_batch(i)
                
                _,dc = sess.run([dis_opt,dloss], feed_dict = {blur: batch_blur, real: np.array(batch_real)})
                _,gc,summary = sess.run([gen_opt,gloss,merged], feed_dict = {blur:np.array(batch_blur), real: np.array(batch_real)})
                

                writer.add_summary(summary,i)
                saver.save(sess,'model',global_step = i)
                i+=1
                   
                print("discriminator cost: ",dc)
                print("generator cost: ",gc)
        writer.close()



train()        

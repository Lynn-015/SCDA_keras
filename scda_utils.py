import tensorflow as tf
import pdb

def select_aggregate(feat_map, resize=None):
    A = tf.reduce_sum(feat_map, axis=-1, keepdims=True)
    a = tf.reduce_mean(A, axis=[1, 2], keepdims=True)
    M = tf.to_float(A>a)
    if resize != None:
        M = tf.image.resize_images(M, resize)
    return M

def scda_plus(map1, map2, alpha=1.0):
    _, h1, w1, _ = map1.shape
    
    M1 = select_aggregate(map1)
    M2 = select_aggregate(map2)

    S2 = map2 * M2
    pavg2 = 1.0 / tf.reduce_sum(M2, axis=[1, 2]) * tf.reduce_sum(S2, axis=[1, 2]) # (b, d)
    pmax2 = tf.reduce_max(S2, axis=[1, 2]) # (b, d)
    S2 = tf.concat([pavg2, pmax2], axis=-1) # (b, 2d)

    # upsampling
    M2 = tf.image.resize_images(M2, [h1, w1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    S1 = map1 * (M1 * M2)
    pavg1 = 1.0 / tf.reduce_sum(M1, axis=[1, 2]) * tf.reduce_sum(S1, axis=[1, 2])
    pmax1 = tf.reduce_max(S1, axis=[1, 2]) 
    S1 = tf.concat([pavg1, pmax1], axis=-1)

    Splus = tf.concat([S2, S1*alpha], axis=-1) # (b, 4d)
    Splus = tf.nn.l2_normalize(Splus, 0)

    return Splus

def flip_plus(map1, map2):
    flip1 = tf.image.flip_up_down(map1)
    flip2 = tf.image.flip_up_down(map2)
    return tf.concat([scda_plus(map1, map2), scda_plus(flip1, flip2)], axis=-1) # (b, 8d)

def post_processing(feat, dim=512):
    s, u, v = tf.svd(feat) 
    feat_svd = tf.transpose(v[:dim, :]) # (b, dim)?
    return feat_svd

def scda_flip_plus(maps):
    return post_processing(flip_plus(maps[0], maps[1]))




    
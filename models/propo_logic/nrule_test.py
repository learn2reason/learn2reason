import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
import pickle
import tensorflow as tf
from tf_recurnn_rlnn import value_NN0, value_NN1, value_NN2, value_NN3, value_NN4
np.set_printoptions(linewidth=200)

docs = {}
docs['0'] = 0
docs['1'] = 1
docs['and']=2
docs['or'] =3
docs['not']=4
docs['('] = 5
docs[')'] = 6

n0 = [0,0,0,0,0,0,0] # place holder
n1 = [1,0,0,0,0,0,0] # 0
n2 = [0,1,0,0,0,0,0] # 1
n3 = [0,0,1,0,0,0,0] # and
n4 = [0,0,0,1,0,0,0] # or
n5 = [0,0,0,0,1,0,0] # ~
n6 = [0,0,0,0,0,1,0] # (
n7 = [0,0,0,0,0,0,1] # )
one_hots = [n1,n2,n3,n4,n5,n6,n7]
leaves = ['0', '1', 'and', 'or', 'not', '(', ')',\
        '( 0', '( 1', '0 )', '1 )',\
        '0 and', '0 or', '1 and', '1 or',\
        'and 0', 'or 0', 'and 1', 'or 1', 'error!']
erroridx = leaves.index('error!')
ncombs = len(leaves)

one_hots = []
for i in range(ncombs):
    ni = np.zeros(ncombs)
    ni[i] = 1
    one_hots.append(ni)

def one_hot_preprocess(xs):
    new_xs = []
    for x in xs:
        new_xs.append(one_hots[x])
    return new_xs

nsymbols = 7
nvalue = ncombs
nprece = 4

lr1 = 1e-3
init_v = 1e-2

def read_file(fname):
    xs = []
    ys = []
    cs = []
    for line in open(fname):
        line = line[:-1]
        x = []
        for word in line.split(' '):
            if word == '':
                continue
            try:
                x.append(docs[word])
            except:
                print('not parsable',line, word)
                sys.exit()
        if '( )' in line or '0 (' in line or '1 (' in line or ') (' in line:
            ys.append(erroridx)
            # continue
        else:
            try:
                ys.append(int(eval(line)))
                print('correct', line)
            except:
                ys.append(erroridx)
                # continue
        xs.append(x)
        cs.append(line)
    return (xs,ys ,cs)

def get_samples():
    new_tx = []
    new_ty = []
    new_tc = []
    new_er = []
    for i in range(len(leaves)):
        if i == erroridx:
            continue
        for j in range(len(leaves)):
            if j == erroridx:
                continue
            li = leaves[i]
            lj = leaves[j]
            newc = li+' '+lj
            if '( )' in newc or '0 (' in newc or '1 (' in newc or ') (' in newc:
                new_er.append([i,j])
                continue
            else:
                try:
                    new_ty.append(int(eval(newc)))
                except:
                    if newc in leaves:
                        new_ty.append(leaves.index(newc))
                    else:
                        new_er.append([i,j])
                        continue
            new_tx.append([i,j])
            new_tc.append(newc)
    return new_tx, new_ty, new_tc, new_er


class value_NN(object):

    def __init__(self, init_v, v_epsilon=1e-9):
        self.W_s1 = tf.get_variable("W_s1",[2*ncombs+nvalue, 64],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s1 = tf.get_variable("b_s1",[64,],initializer=tf.random_uniform_initializer(0,0))
        self.W_s2 = tf.get_variable("W_s2",[64,2],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s2 = tf.get_variable("b_s2",[2],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(None, 2*ncombs+nvalue))
        self.targets = tf.placeholder(tf.float32,  shape=(None, 2))
        
        self.params = [self.W_s1, self.b_s1, self.W_s2, self.b_s2]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s1) + self.b_s1
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s2) + self.b_s2
        return score

def train_value(training_set, value_option):
    batch_size = 64
    sess = tf.Session()
    with tf.variable_scope('value'):
        if value_option == 0:
            v_nn = value_NN0(init_v, [16], nvalue)
        elif value_option == 1:
            v_nn = value_NN1(init_v, [], nvalue)
        elif value_option == 2:
            v_nn = value_NN2(init_v, [64, 32, 16], nvalue)
        elif value_option == 3:
            v_nn = value_NN3(init_v, [64, 32, 16, 16, 8, 8, 4], nvalue)
        elif value_option == 4:
            v_nn = value_NN4(init_v, [ncombs,ncombs+nvalue,224], nvalue)
        elif value_option == 5:
            v_nn = value_NN0(init_v, [64], nvalue)
        elif value_option == 6:
            v_nn = value_NN0(init_v, [8], nvalue)
        else:
            return -1

        v_logits = v_nn.next_score()
        v_predictions = tf.argmax(v_logits, 1)
        v_labels = tf.argmax(v_nn.targets, 1)
        v_ac = tf.metrics.accuracy(labels=v_labels, predictions = v_predictions)
        v_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=v_nn.targets, logits=v_logits))
        v_train_op = tf.train.AdamOptimizer(lr1).minimize(v_loss, var_list=v_nn.params)

        init = tf.initialize_all_variables()
        init_l = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)
        # # value_model_vars = [k for k in all_vars if k.name.startswith("value")]
        # saver = tf.train.Saver()
        # checkpoint = tf.train.get_checkpoint_state("value_saved_networks")
        # if checkpoint and checkpoint.model_checkpoint_path:
        #     saver.restore(sess, checkpoint.model_checkpoint_path)
        #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
        # else:
        #     print("Could not find old network weights")
        #     return
        saver = tf.train.Saver()

        training_x, training_y, cs, vs, split = training_set
        tx1 = []
        ty1 = []
        tx2 = []
        ty2 = []
        for i in range(len(training_y)):
            # if training_x[i][2*ncombs+erroridx] == 1 and np.argmax(training_y[i]) == 1:
            if np.argmax(training_y[i]) == 1:
                tx1.append(training_x[i])
                ty1.append(training_y[i])
            else:
                tx2.append(training_x[i])
                ty2.append(training_y[i])
        print('tx1', len(tx1), 'tx2', len(tx2))
        # sys.exit()
        idx1 = 0
        idx2 = 0
        tx1 = np.asarray(tx1)
        ty1 = np.asarray(ty1)
        tx2 = np.asarray(tx2)
        ty2 = np.asarray(ty2)
        idxs1 = np.arange(len(tx1), dtype=np.int32)
        np.random.shuffle(idxs1)
        idxs2 = np.arange(len(tx2), dtype=np.int32)
        np.random.shuffle(idxs2)
        for e in range(int(1e7)):
            i1 = idxs1[idx1:idx1+batch_size/2]
            i2 = idxs2[idx2:idx2+batch_size/2]
            x1 = tx1[i1]
            y1 = ty1[i1]
            x2 = tx2[i2]
            y2 = ty2[i2]
            idx1 += batch_size/2
            idx2 += batch_size/2
            if idx1 > len(tx1):
                idx1 = 0
                idxs1 = np.arange(len(tx1), dtype=np.int32)
                np.random.shuffle(idxs1)
            if idx2 > len(tx2):
                idx2 = 0
                idxs2 = np.arange(len(tx2), dtype=np.int32)
                np.random.shuffle(idxs2)
            # x = x1 + x2
            # y = y1 + y2
            x = np.concatenate([x1,x2],axis=0)
            y = np.concatenate([y1,y2],axis=0)
            rv_logits, rv_loss, _ = sess.run((v_logits, v_loss, v_train_op), {v_nn.e_in:x, v_nn.targets: y})
            accuracy, rv_labels, rv_pred= sess.run((v_ac, v_labels, v_predictions), {v_nn.e_in:x, v_nn.targets: y})
            nc = 0
            for i in range(len(rv_labels)):
                if rv_labels[i] != rv_pred[i]:
                    nc+= 1
            if e % 10 == 0:
                print('e', e, 'accuracy', accuracy, nc)

            # if nc == 0 and e > 10000 and e % 10 == 0:
            if nc == 0 and e % 50 == 0:
                tnc = 0
                for i in range(len(training_y)):
                    x = training_x[i:i+1]
                    y = training_y[i:i+1]
                    rv_pred, rv_logits = sess.run((v_predictions, v_logits), {v_nn.e_in:x})
                    if np.argmax(y) != rv_pred:
                        tnc += 1
                        print(cs[i], vs[i], np.argmax(y), rv_pred[0], rv_logits[0])
                print('e', e, 'tnc', tnc)
                if tnc == 0:
                    print('train success!')
                    break
        saver.save(sess, 'value_saved_networks/value-dqn'+str(value_option), global_step = e)    
        sess.close()
        # tf.reset_default_graph()
        return e

def gen_data_and_train_value(value_option):
    # training
    tx, ty, tc, terr = get_samples()
    tp = []
    # for i in range(len(tx)):
    #     print(tx[i], ty[i], tc[i])
    training_x = []
    training_y = []
    vs = []
    cs = []
    for i in range(len(tx)):
        for j in range(nvalue):
            x = np.concatenate([one_hots[tx[i][0]], one_hots[tx[i][1]]])
            v = np.zeros(nvalue) 
            v[j] = 1
            training_x.append(np.concatenate([x, v]))
            vs.append(j)
            cs.append(tc[i])
            if j == ty[i]:
                # print('succ')
                training_y.append(np.asarray([0,1]))
            else:
                training_y.append(np.asarray([1,0]))
    for i in range(len(terr)):
        for j in range(nvalue):
            x = np.concatenate([one_hots[terr[i][0]], one_hots[terr[i][1]]])
            v = np.zeros(nvalue) 
            v[j] = 1
            training_x.append(np.concatenate([x, v]))
            vs.append(j)
            cs.append(leaves[terr[i][0]]+' '+leaves[terr[i][1]])
            if j == erroridx:
                # print('succ')
                training_y.append(np.asarray([0,1]))
            else:
                training_y.append(np.asarray([1,0]))
    # for i in range(len(training_y)):
    #     print(cs[i], vs[i], training_y[i])
    print(len(training_y))
    e = train_value([training_x, training_y, cs, vs, len(tx)], value_option)
    return e

class prece_NN(object):

    def __init__(self, init_v, v_epsilon=1e-9):
        self.W_s1 = tf.get_variable("W_s1",[2*ncombs+nprece, 64],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s1 = tf.get_variable("b_s1",[64,],initializer=tf.random_uniform_initializer(0,0))
        self.W_s2 = tf.get_variable("W_s2",[64,2],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s2 = tf.get_variable("b_s2",[2],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(None, 2*ncombs+nprece))
        self.targets = tf.placeholder(tf.float32,  shape=(None, 2))
        
        self.params = [self.W_s1, self.b_s1, self.W_s2, self.b_s2]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s1) + self.b_s1
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s2) + self.b_s2
        return score

def train_precedence(training_set, prece_option):
    batch_size = 32
    # with tf.Session() as sess:
    sess = tf.Session()
    with tf.variable_scope('prece'):
        # s_nn = prece_NN(init_v)
        if prece_option == 0:
            s_nn = value_NN0(init_v, [16], nprece)
        elif prece_option == 1:
            s_nn = value_NN1(init_v, [], nprece)
        elif prece_option == 2:
            s_nn = value_NN2(init_v, [64, 32, 16], nprece)
        elif prece_option == 3:
            s_nn = value_NN3(init_v, [64, 32, 16, 16, 8, 8, 4], nprece)
        elif prece_option == 4:
            s_nn = value_NN4(init_v, [ncombs, ncombs+nprece,96], nprece)
        elif prece_option == 5:
            s_nn = value_NN0(init_v, [64], nprece)
        elif prece_option == 6:
            s_nn = value_NN0(init_v, [8], nprece)
        else:
            return -1

        s_logits = s_nn.next_score()
        s_predictions = tf.argmax(s_logits, 1)
        s_labels = tf.argmax(s_nn.targets, 1)
        s_ac = tf.metrics.accuracy(labels=s_labels, predictions = s_predictions)
        s_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=s_nn.targets, logits=s_logits))
        s_train_op = tf.train.AdamOptimizer(lr1).minimize(s_loss, var_list=s_nn.params)

        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        init_l = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)

        training_x, training_y, cs, ps = training_set
        idxs = np.arange(len(training_y), dtype=np.int32)
        np.random.shuffle(idxs)
        training_x = np.asarray(training_x)
        training_y = np.asarray(training_y)
        idx = 0
        for e in range(int(1e7)):
            ii = idxs[idx:idx+batch_size]
            x = training_x[ii]
            y = training_y[ii]
            idx += batch_size
            if idx > len(training_x):
                idxs = np.arange(len(training_y), dtype=np.int32)
                np.random.shuffle(idxs)
                idx = 0
            rs_logits, rs_loss, _ = sess.run((s_logits, s_loss, s_train_op), {s_nn.e_in:x, s_nn.targets: y})
            accuracy, rs_labels, rs_pred= sess.run((s_ac, s_labels, s_predictions), {s_nn.e_in:x, s_nn.targets: y})
            nc = 0
            for i in range(len(rs_labels)):
                if rs_labels[i] != rs_pred[i]:
                    nc+= 1
            if e % 10 == 0:
                print('e', e, 'accuracy', accuracy, nc)

            if nc == 0:
                tnc = 0
                for i in range(len(training_y)):
                    x = training_x[i:i+1]
                    y = training_y[i:i+1]
                    rs_pred, rs_logits = sess.run((s_predictions, s_logits), {s_nn.e_in:x})
                    if np.argmax(y) != rs_pred:
                        tnc += 1
                        print(cs[i], ps[i], np.argmax(y), rs_pred[0], rs_logits[0])
                print('e', e, 'tnc', tnc)
                if tnc == 0:
                    print('train success!')
                    break
        saver.save(sess, 'prece_saved_networks/precedence-dqn'+str(prece_option), global_step = e)    
        sess.close()
        # tf.reset_default_graph()
        return e

def gen_data_and_train_precedence(prece_option):
    # training
    tx, ty, tc, _ = get_samples()
    tp = []
    # for i in range(len(tx)):
    #     print(tx[i], ty[i], tc[i])
    training_x = []
    training_y = []
    ps = []
    cs = []
    for i in range(len(tx)):
        for j in range(nprece):
            x = np.concatenate([one_hots[tx[i][0]], one_hots[tx[i][1]]])
            p = np.zeros(nprece) 
            p[j] = 1
            training_x.append(np.concatenate([x, p]))
            tp = 0
            if 'not' in tc[i]:
                tp = 3
            elif 'and' in tc[i]:
                tp = 2
            elif 'or' in tc[i]:
                tp = 1
            else:
                tp = 0
            ps.append(j)
            cs.append(tc[i])
            # print(j, tp)
            if j == tp:
                # print('succ')
                training_y.append(np.asarray([0,1]))
            else:
                training_y.append(np.asarray([1,0]))
    # for i in range(len(training_y)):
    #     print(cs[i])
    print(len(training_y))
    e = train_precedence([training_x, training_y, cs, ps], prece_option)
    return e

def test(real_examples, mode, option):
    print('init')
    sess = tf.InteractiveSession() 
    with tf.variable_scope('prece'):
        # s_nn = value_NN3(init_v, [64, 32, 16, 16, 8, 8, 4], nprece)
        # s_nn = prece_NN(init_v, nprece)

        if option == 0:
            s_nn = value_NN0(init_v, [16], nprece)
        elif option == 1:
            s_nn = value_NN1(init_v, [], nprece)
        elif option == 2:
            s_nn = value_NN2(init_v, [64, 32, 16], nprece)
        elif option == 3:
            s_nn = value_NN3(init_v, [64, 32, 16, 16, 8, 8, 4], nprece)
        elif option == 4:
            s_nn = value_NN4(init_v, [ncombs, ncombs+nprece,96], nprece)
        elif option == 5:
            s_nn = value_NN0(init_v, [64], nprece)
        elif option == 6:
            s_nn = value_NN0(init_v, [8], nprece)
        else:
            print('error', option)
            return -1

        s_logits = tf.nn.softmax(s_nn.next_score())
        s_predictions = tf.argmax(s_logits, 1)
        s_labels = tf.argmax(s_nn.targets, 1)
        s_ac = tf.metrics.accuracy(labels=s_labels, predictions = s_predictions)

    print('init')
    with tf.variable_scope('value'):
        # v_nn = value_NN3(init_v, [64, 32, 16, 16, 8, 8, 4], nvalue)
        # v_nn = value_NN(init_v,  nvalue)
        if option == 0:
            v_nn = value_NN0(init_v, [16], nvalue)
        elif option == 1:
            v_nn = value_NN1(init_v, [], nvalue)
        elif option == 2:
            v_nn = value_NN2(init_v, [64, 32, 16], nvalue)
        elif option == 3:
            v_nn = value_NN3(init_v, [64, 32, 16, 16, 8, 8, 4], nvalue)
        elif option == 4:
            v_nn = value_NN4(init_v, [ncombs,ncombs+nvalue,224], nvalue)
        elif option == 5:
            v_nn = value_NN0(init_v, [64], nvalue)
        elif option == 6:
            v_nn = value_NN0(init_v, [8], nvalue)
        else:
            print('error', option)
            return -1

        v_logits = tf.nn.softmax(v_nn.next_score())
        v_predictions = tf.argmax(v_logits, 1)
        v_labels = tf.argmax(v_nn.targets, 1)
        v_ac = tf.metrics.accuracy(labels=v_labels, predictions = v_predictions)
        # v_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=v_nn.targets, logits=v_logits))
        # v_train_op = tf.train.AdamOptimizer(lr1).minimize(v_loss, var_list=v_nn.params)
    print('init')

    all_vars = tf.all_variables()
    value_model_vars = [k for k in all_vars if k.name.startswith("value")]
    prece_model_vars = [k for k in all_vars if k.name.startswith("prece")]
    vsaver = tf.train.Saver(value_model_vars)
    psaver = tf.train.Saver(prece_model_vars)
    checkpoint = tf.train.get_checkpoint_state("good_value_saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        vsaver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
        return
    checkpoint = tf.train.get_checkpoint_state("good_prece_saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        psaver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
        return

    correct = 0
    for reals in real_examples:
        xs, y, cs = reals
        cal = recurnn_eval(xs, sess, s_predictions, s_nn, v_predictions, v_nn, s_logits, v_logits)
        if np.argmax(cal) == y:
            print('correct', cs, y)
            correct += 1
        else:
            print('error', cs, y, cal)
            # return
    # print('correct', len(real_examples))
    fresult = open('cresult.txt', 'a')
    fresult.write('{0} {1} correct {2} all {3} \n'.format(mode, option, correct, len(real_examples)))
    fresult.close()

def recurnn_eval(xs, sess, s_pred, s_nn, v_pred, v_nn, s_logits, v_logits):
    errorflag = False
    nt = xs[:]
    while len(nt) > 1:

        nnt = []
        scores = []
        for i in range(0, len(nt)-1):
            nnt0 = np.zeros(ncombs)
            ne = -1
            rv_logits_s= []
            for v in range(nvalue):
                tv = np.zeros(nvalue)
                tv[v] = 1
                e_in_s = np.concatenate([nt[i], nt[i+1], tv])
                e_in_s = np.expand_dims(e_in_s, axis=0)
                rv_pred, rv_logits= sess.run((v_pred, v_logits), {v_nn.e_in:e_in_s})
                rv_logits_s.append(rv_logits[0][1])
                if rv_pred == 1:
                    ne = v
                    break
            if ne == -1:
                ne = np.argmax(np.asarray(rv_logits_s))
                # print('error')
                # errorflag = True
                # break

            nnt0[ne] = 1
            nnt.append(nnt0)

            if np.argmax(nnt0) == erroridx:
                scores.append(0)
            else:
                score = -1
                rs_logits_s= []
                for p in range(nprece):
                    tp = np.zeros(nprece)
                    tp[p] = 1
                    e_in_s = np.concatenate([nt[i], nt[i+1], tp])
                    e_in_s = np.expand_dims(e_in_s, axis=0)
                    rs_pred, rs_logits= sess.run((s_pred, s_logits), {s_nn.e_in:e_in_s})
                    rs_logits_s.append(rs_logits[0][1])
                    if rs_pred == 1:
                        score = p+1
                        break
                if score == -1:
                    score = np.argmax(np.asarray(rs_logits_s)) + 1
                    # print('error')
                    # errorflag = True
                    # break
                scores.append(score)

        if errorflag:
            ent = np.zeros(ncombs)
            ent[erroridx] = 1
            return ent

        max_idx = np.argmax(np.asarray(scores))
        nnt = np.asarray(nnt)
        print('scores', scores, max_idx)
        fnt = []
        for i in range(len(nt)):
            if i == max_idx +1:
                continue
            elif i == max_idx:
                fnt.append(nnt[i])
            else:
                fnt.append(nt[i])
        nt = fnt
    print('final result', nt)
    return nt[0]

if __name__ == '__main__':
    # mode = int(sys.argv[1])
    # option = int(sys.argv[2])
    # f = open('result.txt', 'a')
    # if mode == 1:
    #     pe = gen_data_and_train_precedence(option)
    #     f.write('prece {0} {1}\n'.format(option, pe))
    # elif mode == 2:
    #     ve = gen_data_and_train_value(option)
    #     f.write('value {0} {1}\n'.format(option, ve))
    # else:
    #     print('error mode!')
    # f.close()

    
    # f = open('result.txt', 'w')
    # for prece_option in range(1, 5):
    #     pe = gen_data_and_train_precedence(prece_option)
    #     f.write('prece {0} {1}\n'.format(prece_option, pe))
    # for value_option in range(1, 5):
    #     ve = gen_data_and_train_value(value_option)
    #     f.write('value {0} {1}\n'.format(value_option, ve))
    # f.close()
    
    # # testing
    fname1 = './logic_t4.txt'
    tx, ty, cs = read_file(fname1)
    tx = tx[:1000]
    ty = ty[:1000]
    cs = cs[:1000]
    # tx1, ty1, cs1 = read_file(fname1)
    # tx2, ty2, cs2 = read_file(fname2)
    # tx = tx1 + tx2
    # ty = ty1 + ty2
    # cs = cs1 + cs2
    real_samples = []
    for idx,x in enumerate(tx):
        xs = one_hot_preprocess(x)
        real_samples.append([xs, ty[idx], cs[idx]])

    print(real_samples[0])

    # real_examples = np.random.shuffle(real_examples)
    mode = int(sys.argv[1])
    option = int(sys.argv[2])
    test(real_samples, mode, option)

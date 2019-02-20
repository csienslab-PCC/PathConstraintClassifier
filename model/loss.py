
import math

sr0 = [
    {"time":0.001, "error":False, "answer":'sat'},
    {"time":0.002, "error":False, "answer":'sat'},
    {"time":0.003, "error":False, "answer":'sat'},
    {"time":0.004, "error":False, "answer":'sat'},
    {"time":0.005, "error":False, "answer":'sat'},
    {"time":0.1, "error":True, "answer":''},
    {"time":100, "error":True, "answer":''}
]


sr1 = [
    {"time":1, "error":False, "answer":'sat'},
    {"time":2, "error":False, "answer":'sat'},
    {"time":3, "error":False, "answer":'sat'},
    {"time":4, "error":False, "answer":'sat'},
    {"time":5, "error":False, "answer":'sat'},
    {"time":0.1, "error":True, "answer":''},
    {"time":100, "error":True, "answer":''}
]

sr2 = [
    {"time":15, "error":False, "answer":'sat'},
    {"time":30, "error":False, "answer":'sat'},
    {"time":45, "error":False, "answer":'sat'},
    {"time":60, "error":False, "answer":'sat'},
    {"time":75, "error":False, "answer":'sat'},
    {"time":0.1, "error":True, "answer":''},
    {"time":100, "error":True, "answer":''}
]

sr3 = [
    {"time":20, "error":False, "answer":'sat'},
    {"time":40, "error":False, "answer":'sat'},
    {"time":60, "error":False, "answer":'sat'},
    {"time":80, "error":False, "answer":'sat'},
    {"time":100, "error":False, "answer":'sat'},
    {"time":0.1, "error":True, "answer":''},
    {"time":100, "error":True, "answer":''}
]

def loss_3(x, max_t, min_t, max_loss=100.0, max_abs_time = 100):
    scale_factor = float(x - min_t) / (max_abs_time - min_t)

    return loss_1(x, max_t, min_t) * scale_factor

def loss_2(x, max_t, min_t, max_loss=100.0, max_abs_time = 100):

    scale_factor = (math.log(x) - math.log(min_t)) / (math.log(max_abs_time) - math.log(min_t))
#    scale_factor = float(x - min_t) / (max_abs_time - min_t)

    return loss_1(x, max_t, min_t) * scale_factor

def loss_1(x, max_t, min_t, max_loss=100.0):

    dist = float(max_t - min_t)

    return (max_loss / 2) * (x - min_t) / dist



def gen_loss_vec(sr, loss_func, timeout=100, max_loss=100):
    
    
    loss = loss_func

    loss_vec = [0 for _ in sr]

    for i, x in enumerate(sr):
        if x['time'] > timeout:
            
            loss_vec[i] += max_loss

        elif x['time'] < timeout and len(x['answer']) == 0:
            
            loss_vec[i] += max_loss

        elif x['error']:
            
            loss_vec[i] += max_loss * 2


    solve = sorted([(i, x['time']) for i, x in enumerate(sr) if x['answer']], key=lambda x: x[1])
    print solve

    if len(solve) >= 2:
        
        max_t = solve[-1][1]
        min_t = solve[0][1]

        for s in solve:
            
            loss_vec[s[0]] += loss(s[1], max_t, min_t, max_loss)
    
    return loss_vec


if __name__ == '__main__':
    
    
    print gen_loss_vec(sr0, loss_1)
    print gen_loss_vec(sr1, loss_1)
    print gen_loss_vec(sr2, loss_1)
    print gen_loss_vec(sr3, loss_1)

    print ''
    print gen_loss_vec(sr0, loss_2)
    print gen_loss_vec(sr1, loss_2)
    print gen_loss_vec(sr2, loss_2)
    print gen_loss_vec(sr3, loss_2)

    print ''
    print gen_loss_vec(sr0, loss_3)
    print gen_loss_vec(sr1, loss_3)
    print gen_loss_vec(sr2, loss_3)
    print gen_loss_vec(sr3, loss_3)



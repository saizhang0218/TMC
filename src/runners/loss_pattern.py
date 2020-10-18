######### generate loss pattern##########
import torch as th

prob_transition_matrix_light = [
    [0.99,0.01,0.0],
    [0.85,0.0,0.15],
    [0.91,0.0,0.09]
] 
prob_transition_matrix_medium = [
    [0.94,0.06,0.0,0.0,0.0],
    [0.6,0.4,0.0,0.0,0.0],
    [0.71,0.29,0.0,0.0,0.0],
    [0.87,0.13,0.5,0.0,0.0],
    [0.97,0.03,0.0,0.0,0.0],
    [0.99,0.01,0.0,0.0,0.0]
]   
prob_transition_matrix_heavy = [
    [0.9,0.1,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.81,0.0,0.19,0.0,0.0,0.0,0.0,0.0],
    [0.45,0.0,0.0,0.55,0.0,0.0,0.0,0.0],
    [0.61,0.0,0.0,0.0,0.39,0.0,0.0,0.0],
    [0.37,0.0,0.0,0.0,0.0,0.63,0.0,0.0],
    [0.77,0.0,0.0,0.0,0.0,0.0,0.23,0.0],
    [0.89,0.0,0.0,0.0,0.0,0.0,0.0,0.11],
    [0.93,0.0,0.0,0.0,0.0,0.0,0.0,0.07]
]   

def generate_loss_pattern_heavy(episode_length = 200, dimension = 48*6, p_matrix = prob_transition_matrix_heavy):
    rand = th.rand((episode_length,dimension))
    loss_pattern = th.zeros((episode_length,dimension)).cuda()
    for i in range(dimension):
        state = '0'   # '0' means no error, '1' means error
        for ind in range(episode_length):
            if (state == '0') and (rand[ind,i] <= p_matrix[0][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 1
            elif (state == '0') and (rand[ind,i] >= p_matrix[0][0]):
                next_state = '1'
                loss_pattern[ind,i] = 1
            elif (state == '1') and (rand[ind,i] <= p_matrix[1][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '1') and (rand[ind,i] >= p_matrix[1][0]): 
                next_state = '2'
                loss_pattern[ind,i] = 0
            elif (state == '2') and (rand[ind,i] <= p_matrix[2][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '2') and (rand[ind,i] >= p_matrix[2][0]): 
                next_state = '3'
                loss_pattern[ind,i] = 0
            elif (state == '3') and (rand[ind,i] <= p_matrix[3][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '3') and (rand[ind,i] >= p_matrix[3][0]): 
                next_state = '4'
                loss_pattern[ind,i] = 0
            elif (state == '4') and (rand[ind,i] <= p_matrix[4][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '4') and (rand[ind,i] >= p_matrix[4][0]): 
                next_state = '5'
                loss_pattern[ind,i] = 0
            elif (state == '5') and (rand[ind,i] <= p_matrix[5][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '5') and (rand[ind,i] >= p_matrix[5][0]): 
                next_state = '6'
                loss_pattern[ind,i] = 0
            elif (state == '6') and (rand[ind,i] <= p_matrix[6][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '6') and (rand[ind,i] >= p_matrix[6][0]): 
                next_state = '7'
                loss_pattern[ind,i] = 0
            elif (state == '7') and (rand[ind,i] <= p_matrix[7][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '7') and (rand[ind,i] >= p_matrix[7][0]): 
                next_state = '7'
                loss_pattern[ind,i] = 0
            state = next_state
    return loss_pattern    
    

def generate_loss_pattern_medium(episode_length = 200, dimension = 48*6, p_matrix = prob_transition_matrix_medium):
    rand = th.rand((episode_length,dimension))
    loss_pattern = th.zeros((episode_length,dimension)).cuda()
    for i in range(dimension):
        state = '0'   # '0' means error, '1' means no error
        for ind in range(episode_length):
            if (state == '0') and (rand[ind,i] <= p_matrix[0][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 1
            elif (state == '0') and (rand[ind,i] >= p_matrix[0][0]):
                next_state = '1'
                loss_pattern[ind,i] = 1
            elif (state == '1') and (rand[ind,i] <= p_matrix[1][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '1') and (rand[ind,i] >= p_matrix[1][0]): 
                next_state = '2'
                loss_pattern[ind,i] = 0
            elif (state == '2') and (rand[ind,i] <= p_matrix[2][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '2') and (rand[ind,i] >= p_matrix[2][0]): 
                next_state = '3'
                loss_pattern[ind,i] = 0
            elif (state == '3') and (rand[ind,i] <= p_matrix[3][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '3') and (rand[ind,i] >= p_matrix[3][0]): 
                next_state = '4'
                loss_pattern[ind,i] = 0
            elif (state == '4') and (rand[ind,i] <= p_matrix[4][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '4') and (rand[ind,i] >= p_matrix[4][0]): 
                next_state = '5'
                loss_pattern[ind,i] = 0
            elif (state == '5') and (rand[ind,i] <= p_matrix[5][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '5') and (rand[ind,i] >= p_matrix[5][0]): 
                next_state = '5'
                loss_pattern[ind,i] = 0
            state = next_state
    return loss_pattern    
    
def generate_loss_pattern_light(episode_length = 200, dimension = 48*6, p_matrix = prob_transition_matrix_light):
    rand = th.rand((episode_length,dimension))
    loss_pattern = th.zeros((episode_length,dimension)).cuda()
    for i in range(dimension):
        state = '0'   # '0' means error, '1' means no error
        for ind in range(episode_length):
            if (state == '0') and (rand[ind,i] <= p_matrix[0][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 1
            elif (state == '0') and (rand[ind,i] >= p_matrix[0][0]):
                next_state = '1'
                loss_pattern[ind,i] = 1
            elif (state == '1') and (rand[ind,i] <= p_matrix[1][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '1') and (rand[ind,i] >= p_matrix[1][0]): 
                next_state = '2'
                loss_pattern[ind,i] = 0
            elif (state == '2') and (rand[ind,i] <= p_matrix[2][0]): 
                next_state = '0'
                loss_pattern[ind,i] = 0
            elif (state == '2') and (rand[ind,i] >= p_matrix[2][0]): 
                next_state = '2'
                loss_pattern[ind,i] = 0
            state = next_state
    return loss_pattern            
    
def generate_loss(level='light', env = 'basic_mac_6h_vs_8z'):
    if (level == 'none'):
        if (env=='basic_mac_6h_vs_8z'):
            loss = th.ones((400,8*6*6)).cuda()
        elif (env=='basic_mac_3s_vs_4z'):
            loss = th.ones((400,8*3*3)).cuda()
        elif (env=='basic_mac_3s_vs_5z'):
            loss = th.ones((400,8*3*3)).cuda()
        elif (env=='basic_mac_2c_vs_64zg'):
            loss = th.ones((800,8*2*2)).cuda()
        elif (env=='basic_mac_corridor'):
            loss = th.ones((500,8*6*6)).cuda()
        elif (env=='basic_mac_3s5z'):
            loss = th.ones((400,8*8*8)).cuda()
    elif (level=='light'):
        if (env=='basic_mac_6h_vs_8z'):
            loss = generate_loss_pattern_light(episode_length = 400, dimension = 8*6*6, p_matrix = prob_transition_matrix_light)
        elif (env=='basic_mac_3s_vs_4z'):
            loss = generate_loss_pattern_light(episode_length = 400, dimension = 8*3*3, p_matrix = prob_transition_matrix_light)
        elif (env=='basic_mac_3s_vs_5z'):
            loss = generate_loss_pattern_light(episode_length = 400, dimension = 8*3*3, p_matrix = prob_transition_matrix_light)
        elif (env=='basic_mac_2c_vs_64zg'):
            loss = generate_loss_pattern_light(episode_length = 400, dimension = 8*2*2, p_matrix = prob_transition_matrix_light)
        elif (env=='basic_mac_corridor'):
            loss = generate_loss_pattern_light(episode_length = 500, dimension = 8*6*6, p_matrix = prob_transition_matrix_light)
        elif (env=='basic_mac_3s5z'):
            loss = generate_loss_pattern_light(episode_length = 400, dimension = 8*8*8, p_matrix = prob_transition_matrix_light)
    elif (level=='medium'):
        if (env=='basic_mac_6h_vs_8z'):
            loss = generate_loss_pattern_medium(episode_length = 400, dimension = 8*6*6, p_matrix = prob_transition_matrix_medium)
        elif (env=='basic_mac_3s_vs_4z'):
            loss = generate_loss_pattern_medium(episode_length = 400, dimension = 8*3*3, p_matrix = prob_transition_matrix_medium)
        elif (env=='basic_mac_3s_vs_5z'):
            loss = generate_loss_pattern_medium(episode_length = 400, dimension = 8*3*3, p_matrix = prob_transition_matrix_medium)
        elif (env=='basic_mac_2c_vs_64zg'):
            loss = generate_loss_pattern_medium(episode_length = 400, dimension = 8*2*2, p_matrix = prob_transition_matrix_medium)
        elif (env=='basic_mac_corridor'):
            loss = generate_loss_pattern_medium(episode_length = 500, dimension = 8*6*6, p_matrix = prob_transition_matrix_medium)
        elif (env=='basic_mac_3s5z'):
            loss = generate_loss_pattern_medium(episode_length = 400, dimension = 8*8*8, p_matrix = prob_transition_matrix_medium)
    else:
        if (env=='basic_mac_6h_vs_8z'):
            loss = generate_loss_pattern_heavy(episode_length = 400, dimension = 8*6*6, p_matrix = prob_transition_matrix_heavy)
        elif (env=='basic_mac_3s_vs_4z'):
            loss = generate_loss_pattern_heavy(episode_length = 400, dimension = 8*3*3, p_matrix = prob_transition_matrix_heavy)
        elif (env=='basic_mac_3s_vs_5z'):
            loss = generate_loss_pattern_heavy(episode_length = 400, dimension = 8*3*3, p_matrix = prob_transition_matrix_heavy)
        elif (env=='basic_mac_2c_vs_64zg'):
            loss = generate_loss_pattern_heavy(episode_length = 400, dimension = 8*2*2, p_matrix = prob_transition_matrix_heavy)
        elif (env=='basic_mac_corridor'):
            loss = generate_loss_pattern_heavy(episode_length = 500, dimension = 8*6*6, p_matrix = prob_transition_matrix_heavy)
        elif (env=='basic_mac_3s5z'):
            loss = generate_loss_pattern_heavy(episode_length = 400, dimension = 8*8*8, p_matrix = prob_transition_matrix_heavy)
    return loss
    

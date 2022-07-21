from torchsummary import summary
from utils import get_random_loc, get_CONF, get_json_comp_conf
import torch
from NeuralNet import Net


def get_neural_model(CONF, json_comp_conf):
    weight_file = "depoeharbor_41x41_epoch1_oracle_r4_t1100_s50_rollout:True_batch128"
    # weight_file = "depoeharbor_41x41_epoch1_oracle_r4_t400_s80_rollout:True_batch128"
    # weight_file = "circularharbor_41x41_epoch1_oracle_r4_t1200_s40_rollout:True_batch128_senserange7"
    print("weight_file for network: ", weight_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)
    neural_model = Net().to(device)
    neural_model.load_state_dict(torch.load(
        CONF[json_comp_conf]["neural_net_weights_path"]+weight_file))
    neural_model.eval()

    return neural_model, device

if __name__ == "__main__":

    CONF = get_CONF()
    json_comp_conf = get_json_comp_conf()
    neural_model = get_neural_model(CONF, json_comp_conf)
    device = neural_model[1]

    model, device = get_neural_model(CONF, json_comp_conf)

    summary(model, (4, 41, 41))

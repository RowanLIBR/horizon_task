clear

%hyper params

global use_ppbs
use_ppbs = 1;
struct_length = 80;
global exhaustive;
filename = "EIT_horizontask_sub1_20181003T084055";
exhaustive = 0;
global c;
import_data(filename)
future_const = 1;
c = 1;
hard_coded_example = 1;
learning_rate = 2;  
bucket_size = 1;
num_trials = 80;
gambling_bias = 1;
global precision;
global epi_c;
global nov_c;
mean_difference = 10;
epi_c = 1;
nov_c = 1;
precision = 1/200;
%%%%
global correct_choice;

for epoch = 1:1

for trial = 1:struct_length
counter = 1;
params = load_parameters(game, trial);
%[left better, right better]
d{1} = [1, 1];
horizon = params{1};

%----------observations----------

% observation 1 - last right number

% observation 2 - last left number


%--------Likelihoods-------------
% for the generative process, the likelihood is just the true gausian 
% A{1} = A(Left Mean, 8)
% A{2} = A(Right Mean, 8)

% for the generative model, the likelihood is the constructed gausian from
% the inferred mean and std dev
cc = zeros(100,horizon);
a{1} = ones(1,100/bucket_size)/1000; % left mean
a{2} = ones(1,100/bucket_size)/1000; % right mean
length = numel(a{1});
a{1}(length/2:length) = a{1}(length/2:length)*gambling_bias;
a{2}(length/2:length) = a{2}(length/2:length)*gambling_bias;
std_dev = 8;
obs_right = [];
obs_left = [];
x = [1:1:100];
t = 0;
auto = 1;
hardcoded_left = params{2};
hardcoded_right = params{3};
equal = params{4};
if hard_coded_example == 0
    [left_mean, right_mean] = getMeans(mean_difference);
else
     left_mean = params{7};
     right_mean = params{8};
end

if left_mean > right_mean
    correct_choice = 1;
else
    correct_choice = 2;
end

while t <= horizon
    % for first 4 rounds, the game is played automatically, either with
    % equal or unequal representation from both choices
    if auto == 1
        if equal == 1
            if hard_coded_example == 0
                [a{1}, obs_left, left_1] = sample_and_update(left_mean, std_dev, bucket_size, a{1}, learning_rate, obs_left);
                [a{1}, obs_left, left_2] = sample_and_update(left_mean, std_dev, bucket_size, a{1}, learning_rate, obs_left);
                [a{2}, obs_right, right_1] = sample_and_update(right_mean, std_dev, bucket_size, a{2}, learning_rate, obs_right);
                [a{2}, obs_right, right_2] = sample_and_update(right_mean, std_dev, bucket_size, a{2}, learning_rate, obs_right);
            else
                [a{1}, obs_left, left_1, counter] = hardcoded_number(hardcoded_left,bucket_size, a{1}, learning_rate, obs_left, counter);
                [a{1}, obs_left, left_2, counter] = hardcoded_number(hardcoded_left,bucket_size, a{1}, learning_rate, obs_left, counter);
                [a{2}, obs_right, right_1, counter] = hardcoded_number(hardcoded_right,bucket_size, a{2}, learning_rate, obs_right, counter);
                [a{2}, obs_right, right_2, counter] = hardcoded_number(hardcoded_right,bucket_size, a{2}, learning_rate, obs_right, counter);
            end
            estimate_left_mean = (left_1 + left_2)/2;
            estimate_right_mean = (right_1 + right_2)/2;
            if estimate_left_mean > estimate_right_mean
                d{1}(1) = d{1}(1) + learning_rate;
            else
                d{1}(2) = d{1}(2) + learning_rate;
            end
               
        else
            if hard_coded_example == 1
                if params{5} == 1
                    [a{1}, obs_left, left_1, counter] = hardcoded_number(hardcoded_left,bucket_size, a{1}, learning_rate, obs_left, counter);
                    [a{1}, obs_left, left_2, counter] = hardcoded_number(hardcoded_left,bucket_size, a{1}, learning_rate, obs_left, counter);
                    [a{1}, obs_left, left_3, counter] = hardcoded_number(hardcoded_left,bucket_size, a{1}, learning_rate, obs_left, counter);
                    [a{2}, obs_right, right_1, counter] = hardcoded_number(hardcoded_right,bucket_size, a{2}, learning_rate, obs_right, counter);
                    estimate_left_mean = (left_1 + left_2 + left_3)/3;
                    estimate_right_mean = right_1;
                    if estimate_left_mean > estimate_right_mean
                        d{1}(1) = d{1}(1) + learning_rate;
                    else
                        d{1}(2) = d{1}(2) + learning_rate;
                    end
                else
                    [a{1}, obs_left, left_1, counter] = hardcoded_number(hardcoded_left,bucket_size, a{1}, learning_rate, obs_left, counter);
                    [a{2}, obs_right, right_1, counter] = hardcoded_number(hardcoded_right,bucket_size, a{2}, learning_rate, obs_right, counter);
                    [a{2}, obs_right, right_2, counter] = hardcoded_number(hardcoded_right,bucket_size, a{2}, learning_rate, obs_right, counter);
                    [a{2}, obs_right, right_3, counter] = hardcoded_number(hardcoded_right,bucket_size, a{2}, learning_rate, obs_right, counter);
                    estimate_left_mean = (left_1 + left_2 + left_3)/3;
                    estimate_right_mean = right_1;
                    if estimate_left_mean > estimate_right_mean
                        d{1}(1) = d{1}(1) + learning_rate;
                    else
                        d{1}(2) = d{1}(2) + learning_rate;
                    end
                end
            else
                side = randsample(2,1);
                if side == 1
                    [a{1}, obs_left, left_1] = sample_and_update(left_mean, std_dev, bucket_size, a{1}, learning_rate, obs_left);
                    [a{1}, obs_left, left_2] = sample_and_update(left_mean, std_dev, bucket_size, a{1}, learning_rate, obs_left);
                    [a{1}, obs_left, left_3] = sample_and_update(left_mean, std_dev, bucket_size, a{1}, learning_rate, obs_left);
                    [a{2}, obs_right, right_1] = sample_and_update(right_mean, std_dev, bucket_size, a{2}, learning_rate, obs_right);

                    estimate_left_mean = (left_1 + left_2 + left_3)/3;
                    estimate_right_mean = right_1;
                    if estimate_left_mean > estimate_right_mean
                        d{1}(1) = d{1}(1) + learning_rate;
                    else
                        d{1}(2) = d{1}(2) + learning_rate;
                    end
                else
                    [a{1}, obs_left, left_1] = sample_and_update(left_mean, std_dev, bucket_size, a{1}, learning_rate, obs_left);
                    [a{2}, obs_right, right_1] = sample_and_update(left_mean, std_dev, bucket_size, a{2}, learning_rate, obs_right);
                    [a{2}, obs_right, right_2] = sample_and_update(left_mean, std_dev, bucket_size, a{2}, learning_rate, obs_right);
                    [a{2}, obs_right, right_3] = sample_and_update(right_mean, std_dev, bucket_size, a{2}, learning_rate, obs_right);
                    estimate_right_mean = (right_1 + right_2 + right_3)/3;
                    estimate_left_mean = left_1;
                    if estimate_left_mean > estimate_right_mean
                        d{1}(1) = d{1}(1) + learning_rate;
                    else
                        d{1}(2) = d{1}(2) + learning_rate;
                    end
                end
            end
        
        end
        auto = 0;
        t = 4;
        mean_bucket = ceil(estimate_left_mean/bucket_size);
        a{1}(mean_bucket) = a{1}(mean_bucket) + learning_rate;
        mean_bucket = ceil(estimate_right_mean/bucket_size);
        a{2}(mean_bucket) = a{2}(mean_bucket) + learning_rate;
    else
        % now we actually play the game after the first 4 dummy rounds
        % we start by sampling priors from the dirichlet distributions
        norm_left = normalise(a{1});
        norm_right = normalise(a{2});
        
        left_mean_prior = (find(cumsum(norm_left) >= rand,1))*bucket_size;
        left_mean_prior = randi([left_mean_prior-bucket_size, left_mean_prior],1);
        right_mean_prior = (find(cumsum(norm_right) >= rand,1))*bucket_size;
        right_mean_prior = randi([right_mean_prior-bucket_size, right_mean_prior],1);
        std_prior = round(calc_std(estimate_left_mean, estimate_right_mean, obs_right, obs_left));
        
        
        G = forward_tree_search(left_mean_prior, right_mean_prior, std_prior, a, d, t, obs_left, obs_right, estimate_left_mean, estimate_right_mean, horizon, learning_rate, bucket_size, t);
        [maxi, choice] = max(G);
        choices(trial,t) = choice;
        if t < horizon
        if choice == 1
            if hard_coded_example == 0
                left_1 = max(1,round(normrnd(left_mean, std_dev)));
                left_1 = min(100,left_1);
            else
                [a{1}, obs_left, left_1, counter] = hardcoded_number(hardcoded_left,bucket_size, a{1}, learning_rate, obs_left, counter);
                
            end
            estimate_left_mean = estimate_left_mean + (left_1 - estimate_left_mean)/(numel(obs_left)+1);          
            mean_bucket = ceil(estimate_left_mean/bucket_size);
            obs_left(end+1) = left_1;
            a{1}(mean_bucket) = a{1}(mean_bucket) + learning_rate;
        else
            if hard_coded_example == 0
                right_1 = max(1,round(normrnd(right_mean, std_dev)));
                right_1 = min(100,right_1);
            else
                [a{2}, obs_right, right_1, counter] = hardcoded_number(hardcoded_right,bucket_size, a{2}, learning_rate, obs_right, counter);
            end
            estimate_right_mean = estimate_right_mean + (right_1 - estimate_right_mean)/(numel(obs_right)+1);
            
            mean_bucket = ceil(estimate_right_mean/bucket_size);
            obs_right(end+1) = right_1;
            a{2}(mean_bucket) = a{2}(mean_bucket) + learning_rate;
        end
        if estimate_left_mean > estimate_right_mean
            d{1}(1) = d{1}(1) + learning_rate;
        else
            d{1}(2) = d{1}(2) + learning_rate;
        end
    end
    
    t = t+1;
    
    end
end

end
correct_count = 0;

for i = 5:horizon
    for j = 1:num_trials
        if choices(j,i) == correct_choice
            correct_count = correct_count + 1;
        end
    end
    correct_counts(i) = correct_count/num_trials;
    correct_count = 0;
end
cc(epoch,:) = correct_counts;
end
result = sum(cc,1)/epoch;
function G = forward_tree_search(left_mean, right_mean, std_dev, a, d, t, obs_left, obs_right, estimate_left_mean, estimate_right_mean, horizon, learning_rate, bucket_size, true_t)
    global use_ppbs
    global exhaustive
    global correct_choice
    global precision
    global epi_c;
    global nov_c;
    global c;
    G = [0, 0];
    d_prior = d{1};
    [predicted_obs_left, I] = min([100, round(normrnd(left_mean, std_dev))]);
    [predicted_obs_left, I] = max([1, predicted_obs_left]);
    obs_left(end+1) = predicted_obs_left;
    estimate_left_mean = estimate_left_mean + (predicted_obs_left - estimate_left_mean)/(numel(obs_left)+1);
    if estimate_left_mean > estimate_right_mean
        d{1}(1) = d{1}(1) + learning_rate;
    else
        d{1}(2) = d{1}(2) + learning_rate;
    end
    mean_bucket = ceil(estimate_left_mean/bucket_size);
    G(1) = G(1) + predicted_obs_left*precision;
    
    if t < horizon
        a_prior = a{1};
        if use_ppbs == 1 
            a{1}(mean_bucket) = a{1}(mean_bucket) + learning_rate;   
            novelty = kldir(normalise(a_prior), normalise(a{1}));
        else
            a_temp = a{1};
            a_temp(mean_bucket) = a_temp(mean_bucket) + learning_rate;
            novelty = kldir(normalise(a_prior), normalise(a_temp));
        end     
        epi = kldir(normalise(d_prior), normalise(d{1}));
        G(1) = G(1) + nov_c*novelty + epi_c*epi;
        norm_left = normalise(a{1});
        if exhaustive == 0
            left_mean_prior = (find(cumsum(norm_left) >= rand,1))*bucket_size;
            left_mean_prior = randi([left_mean_prior-bucket_size left_mean_prior],1);
            std_dev = round(calc_std(estimate_left_mean, estimate_right_mean, obs_right, obs_left));
            E = forward_tree_search(left_mean_prior, right_mean, std_dev, a, d, t+1, obs_left, obs_right, estimate_left_mean, estimate_right_mean, horizon, learning_rate, bucket_size, true_t);
            G(1) = G(1) + 0.7 * max(E);
        else
            for mean = 1:numel(a{1})
                left_mean_prior = randi([mean*10-bucket_size mean*10],1);
                std_dev = round(calc_std(estimate_left_mean, estimate_right_mean, obs_right, obs_left));
                E = forward_tree_search(left_mean_prior, right_mean, std_dev, a, d, t+1, obs_left, obs_right, estimate_left_mean, estimate_right_mean, horizon, learning_rate, bucket_size, true_t);
                G(1) = G(1) + a{1}(mean)*max(E);
            end
        end
    end    
    [predicted_obs_right, I] = min([100, round(normrnd(right_mean, std_dev))]);
    [predicted_obs_right, I] = max([1, predicted_obs_right]);
    obs_right(end+1) = predicted_obs_right;
    estimate_right_mean = estimate_right_mean + (predicted_obs_right - estimate_right_mean)/(numel(obs_right)+1);
    if estimate_left_mean > estimate_right_mean
        d{1}(1) = d{1}(1) + learning_rate;
    else
        d{1}(2) = d{1}(2) + learning_rate;
    end
    
    mean_bucket = ceil(estimate_right_mean/bucket_size);
    G(2) = G(2) + predicted_obs_right*precision;
    
    if t < horizon
        a_prior = a{2};
        if use_ppbs == 1 
            a{2}(mean_bucket) = a{2}(mean_bucket) + learning_rate;   
            novelty = kldir(normalise(a_prior), normalise(a{2}));
        else
            a_temp = a{2};
            a_temp(mean_bucket) = a_temp(mean_bucket) + learning_rate;
            novelty = kldir(normalise(a_prior), normalise(a_temp));
        end          
        epi = kldir(normalise(d_prior), normalise(d{1}));
        G(2) = G(2) + nov_c*novelty + epi_c*epi;
        norm_right = normalise(a{2});
        if exhaustive == 0
            right_mean_prior = (find(cumsum(norm_right) >= rand,1))*bucket_size;
            right_mean_prior = randi([right_mean_prior-bucket_size right_mean_prior],1);
            std_dev = round(calc_std(estimate_left_mean, estimate_right_mean, obs_right, obs_left));
            E = forward_tree_search(left_mean, right_mean_prior, std_dev, a, d, t+1, obs_left, obs_right, estimate_left_mean, estimate_right_mean, horizon, learning_rate,bucket_size, true_t);
            G(2) = G(2) + 0.7 * max(E);
        else
            for mean = 1:numel(a{2})
                right_mean_prior = randi([mean*10-bucket_size mean*10],1);
                std_dev = round(calc_std(estimate_left_mean, estimate_right_mean, obs_right, obs_left));
                E = forward_tree_search(left_mean, right_mean_prior, std_dev, a, d, t+1, obs_left, obs_right, estimate_left_mean, estimate_right_mean, horizon, learning_rate, bucket_size, true_t);
                G(1) = G(1) + a{1}(mean)*max(E);
            end
        end
            
    end    
    [maxi, index] = max([G(1),G(2)]);
end
function [mean_1, mean_2] = getMeans(mean_difference)
    sample_1 = [40,60];
    %sample_2 = [4,8,12,20,30];
    mean_1 = randsample(sample_1,1);
    difference = mean_difference;%randsample(sample_2,1);
    x = round(rand(1,1));
    if x == 1
        mean_2 = mean_1 + difference;
    else
        mean_2 = mean_1 - difference;
    end
end

function x = normalise(array)
x = array/(sum(array));
if isnan(x)
    x = ones(numel(x),1);
    x(:) = 1/numel(x);
end
end

function kl = kldir_continuous(sigma_a,mean_a,sigma_b, mean_b)
kl = sigma_b/sigma_a + (sigma_b^2 + (mean_a - mean_b)^2)/2*sigma_b^2 - 1/2;
end

function kl = kldir(a,b)
kl = 0;
for j = 1:numel(a(1,:)) % for each column
    for i = 1:numel(a(:,j)) % for each row
        loga = log(a(i,j));
        logb = log(b(i,j));
       kl =  kl + a(i,j) * (loga - logb);
    end
end
end

function [a, obs, sample] = sample_and_update(mean, std_dev, bucket_size, a, learning_rate, obs)
    sample = max(1,round(normrnd(mean, std_dev)));
    sample = min(100, sample);
%     mean_bucket = ceil(sample/bucket_size);
%     a(mean_bucket) = a(mean_bucket) + learning_rate;
    obs(end+1) = sample;
end
function std = calc_std(left_mean, right_mean, right_obs, left_obs)
    num_left_obs = numel(left_obs);
    num_right_obs = numel(right_obs);
    right_coefficient = num_right_obs/(num_right_obs+num_left_obs);
    left_coefficient = num_left_obs/(num_right_obs+num_left_obs);
    dv = 0;
    for i = 1:num_left_obs
        dv = dv + (left_obs(i) - left_mean)^2;
    end
    
    left_std_dv = left_coefficient*sqrt(dv/num_left_obs);
    dv = 0;
    for i = 1:num_right_obs
        dv = dv + (right_obs(i) - right_mean)^2;
    end
    
    right_std_dv = right_coefficient*sqrt(dv/num_right_obs);
    
    std = left_std_dv + right_std_dv;

    
end

function [a, obs, sample, counter] = hardcoded_number(hard_coded_array, bucket_size, a, learning_rate, obs, counter)
    sample = hard_coded_array(counter);
    counter = counter + 1;
%     mean_bucket = ceil(sample/bucket_size);
%     a(mean_bucket) = a(mean_bucket) + learning_rate;
    obs(end+1) = sample;
end


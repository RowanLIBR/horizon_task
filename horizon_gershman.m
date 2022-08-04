clear

%hyper params
global c;
global b;
b = 0.5;
future_const = 1;
c = 1;
hard_coded_example = 0;
learning_rate = 0.5;
horizon = 10;
equal = 0;  
bucket_size = 5;
num_trials = 500;
gambling_bias = 1;
global tau % reward variance
global precision;
global epi_c;
global nov_c;
global lambda;
mean_difference = 20;
epi_c = 1;
nov_c = 1;
lambda = 1;
gamma = 1;
hard_coded_left_mean = 40;
hard_coded_right_mean = 50;
hardcoded_left = [55, 49, 56, 39, 40, 31, 27, 42, 49, 41];
hardcoded_right = [47, 45, 55, 52, 50, 55, 51, 53, 61, 50];
UCB = 0;
Thompson = 0;
%%%%
N_left = 1;
N_right = 1;
global correct_choice;
if hard_coded_example == 0
    [left_mean, right_mean] = getMeans(mean_difference);
else
     left_mean = hard_coded_left_mean;
     right_mean = hard_coded_right_mean;
end
cc = zeros(100,horizon);
for epoch = 1:1

if left_mean > right_mean
    correct_choice = 1;
else
    correct_choice = 2;
end
for trial = 1:num_trials


%----------observations----------

% observation 1 - last right number

% observation 2 - last left number


%--------Likelihoods-------------
% for the generative process, the likelihood is just the true gausian 
% A{1} = A(Left Mean, 8)
% A{2} = A(Right Mean, 8)

% for the generative model, the likelihood is the constructed gausian from
% the inferred mean and std dev
a{1} = ones(1,100/bucket_size); % left mean
a{2} = ones(1,100/bucket_size); % right mean
length = numel(a{1});
a{1}(length/2:length) = a{1}(length/2:length)*gambling_bias;
a{2}(length/2:length) = a{2}(length/2:length)*gambling_bias;
std_dev = 8;
obs_right = [];
obs_left = [];
x = [1:1:100];
left_posterior = 50;
right_posterior = 50;
q = [100, 100];
t = 0;
auto = 1;
while t <= horizon
    % for first 4 rounds, the game is played automatically, either with
    % equal or unequal representation from both choices
    if auto == 1
        if equal == 1
            if hard_coded_example == 0
                [obs_left, left_posterior, q(1)] = sample_and_update(left_mean, left_posterior, std_dev,q(1), obs_left);
                [obs_left, left_posterior, q(1)] = sample_and_update(left_mean, left_posterior, std_dev,q(1), obs_left);
                [obs_right, right_posterior, q(2)] = sample_and_update(right_mean, right_posterior, std_dev,q(2), obs_right);
                [obs_right, right_posterior, q(2)] = sample_and_update(right_mean, right_posterior, std_dev,q(2), obs_right);
            else
                [obs_left, left_posterior, q(1)] = hardcoded_number(hardcoded_left(1), left_posterior, std_dev,q(1), obs_left);
                [obs_left, left_posterior, q(1)] = hardcoded_number(hardcoded_left(2), left_posterior, std_dev,q(1), obs_left);
                [obs_right, right_posterior, q(2)] = hardcoded_number(hardcoded_right(1), right_posterior, std_dev,q(2), obs_right);
                [obs_right, right_posterior, q(2)] = hardcoded_number(hardcoded_right(2), right_posterior, std_dev,q(2), obs_right);
            end
        else
            if hard_coded_example == 1
                [obs_left, left_posterior, q(1)] = hardcoded_number(hardcoded_left(1), left_posterior, std_dev,q(1), obs_left);
                [obs_left, left_posterior, q(1)] = hardcoded_number(hardcoded_left(2), left_posterior, std_dev,q(1), obs_left);
                [obs_left, left_posterior, q(1)] = hardcoded_number(hardcoded_left(3), left_posterior, std_dev,q(1), obs_left);
                [obs_right, right_posterior, q(2)] = hardcoded_number(hardcoded_right(2), right_posterior, std_dev,q(2), obs_right);
            else
                side = randsample(2,1);
                if side == 1
                    [obs_left, left_posterior, q(1)] = sample_and_update(left_mean,left_posterior, std_dev,q(1), obs_left);
                    [obs_left, left_posterior, q(1)] = sample_and_update(left_mean, left_posterior, std_dev,q(1), obs_left);
                    [obs_left, left_posterior, q(1)] = sample_and_update(left_mean, left_posterior, std_dev,q(1), obs_left);
                    [obs_right, right_posterior, q(2)] = sample_and_update(right_mean, right_posterior, std_dev,q(2), obs_right);
                else
                    [obs_left, left_posterior, q(1)] = sample_and_update(left_mean, left_posterior, std_dev,q(1), obs_left);
                    [obs_right, right_posterior, q(2)] = sample_and_update(left_mean, left_posterior, std_dev,q(2), obs_right);
                    [obs_right, right_posterior, q(2)] = sample_and_update(left_mean, right_posterior, std_dev,q(2), obs_right);
                    [obs_right, right_posterior, q(2)] = sample_and_update(right_mean, right_posterior, std_dev,q(2), obs_right);
                   
                end
            end
        end
        auto = 0;
        t = 4;
        test = 0;
    else
        m = [left_posterior, right_posterior];
        s = [q(1), q(2)];
        if UCB == 1
            v = m + b*sqrt(q);
            p = normcdf((v(1)-v(2))/lambda); 
        elseif Thompson == 1
            p = normcdf((m(1)-m(2))/sqrt(s(1) + s(2)));
        else
            p = normcdf(b*(m(1)-m(2))/(sqrt(s(1)+s(2))) + gamma*(sqrt(s(1))-sqrt(s(2)))); % hybrid between thompson and UCB
        end
        
        if rand < p
            choice = 1;
        else
            choice = 2;
        end
   
        choices(trial,t) = choice;
        if t < horizon
        if choice == 1
            if hard_coded_example == 0
                [obs_left, left_posterior, q] = sample_and_update(left_mean, left_posterior, std_dev, q, obs_left);
            else
                [obs_left, left_posterior, q] = hardcoded_number(left_mean, left_posterior, std_dev, q, obs_left);
            end
        else
            if hard_coded_example == 0
               [obs_right, right_posterior, q] = sample_and_update(right_mean, right_posterior, std_dev, q, obs_right);
            else
                [obs_right, right_posterior, q] = hardcoded_number(hardcoded_right(2), right_posterior, std_dev, q, obs_right);
            end
        end
        end
    end
    
    t = t+1;
    
    
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

function [obs, posterior_mean, std_dev_prior] = sample_and_update(true_mean, posterior_mean, std_dev, std_dev_prior, obs)
    sample = max(1,round(normrnd(true_mean, std_dev)));
    kalman_gain = std_dev_prior/(std_dev_prior + std_dev);
    sample = min(100, sample);
    obs(end+1) = sample;
    posterior_mean = posterior_mean + kalman_gain*(sample - posterior_mean);
    std_dev_prior = std_dev_prior - kalman_gain*std_dev_prior;
    
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

function [obs, posterior_mean, std_dev_prior] = hardcoded_number(number, posterior_mean, std_dv, std_dev_prior, obs)
    kalman_gain = std_dev_prior/(std_dv + std_dev_prior);
    sample = number;
    posterior_mean = posterior_mean + kalman_gain*(sample - posterior_mean);
    std_dev_prior = std_dev_prior - kalman_gain*std_dev_prior;
    obs(end+1) = sample;
end

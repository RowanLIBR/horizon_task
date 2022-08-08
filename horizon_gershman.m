clear

%hyper params
global c;
global b;
b = 0.5;
future_const = 1;
c = 1;
filename = "EIT_horizontask_sub1_20181003T084055";
import_data(filename)
struct_length = 80;
hard_coded_example = 1;
learning_rate = 0.5;

num_trials = 80;
gambling_bias = 1;
global tau % reward variance
global precision;
global epi_c;
global nov_c;
global lambda;
mean_difference = 10;
lambda = 1;
gamma = 1;
UCB = 0;
Thompson = 0;
%%%%
global correct_choice;
for epoch = 1:1
for trial = 1:struct_length
    params = load_parameters(game, trial);


%----------observations----------

% observation 1 - last right number

% observation 2 - last left number


%--------Likelihoods-------------
% for the generative process, the likelihood is just the true gausian 
% A{1} = A(Left Mean, 8)
% A{2} = A(Right Mean, 8)

% for the generative model, the likelihood is the constructed gausian from
% the inferred mean and std dev

std_dev = 8;
obs_right = [];
obs_left = [];
x = [1:1:100];
left_posterior = 50;
right_posterior = 50;
q = [100, 100];
t = 0;
auto = 1;
horizon = params{1};
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
cc = zeros(100,horizon);
counter = 1;
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
                [obs_left, left_posterior, q(1), counter] = hardcoded_number(hardcoded_left, left_posterior, std_dev,q(1), obs_left, counter);
                [obs_left, left_posterior, q(1), counter] = hardcoded_number(hardcoded_left, left_posterior, std_dev,q(1), obs_left, counter);
                [obs_right, right_posterior, q(2), counter] = hardcoded_number(hardcoded_right, right_posterior, std_dev,q(2), obs_right, counter);
                [obs_right, right_posterior, q(2), counter] = hardcoded_number(hardcoded_right, right_posterior, std_dev,q(2), obs_right, counter);
            end
        else
            if hard_coded_example == 1
                if params{5} == 1
                    [obs_left, left_posterior, q(1), counter] = hardcoded_number(hardcoded_left, left_posterior, std_dev,q(1), obs_left, counter);
                    [obs_left, left_posterior, q(1), counter] = hardcoded_number(hardcoded_left, left_posterior, std_dev,q(1), obs_left, counter);
                    [obs_left, left_posterior, q(1), counter] = hardcoded_number(hardcoded_left, left_posterior, std_dev,q(1), obs_left, counter);
                    [obs_right, right_posterior, q(2), counter] = hardcoded_number(hardcoded_right, right_posterior, std_dev,q(2), obs_right, counter);
                else
                    [obs_left, left_posterior, q(1), counter] = hardcoded_number(hardcoded_left, left_posterior, std_dev,q(1), obs_left, counter);
                    [obs_right, right_posterior, q(2), counter] = hardcoded_number(hardcoded_right, right_posterior, std_dev,q(2), obs_right, counter);
                    [obs_right, right_posterior, q(2), counter] = hardcoded_number(hardcoded_right, right_posterior, std_dev,q(2), obs_right, counter);
                    [obs_right, right_posterior, q(2), counter] = hardcoded_number(hardcoded_right, right_posterior, std_dev,q(2), obs_right, counter);
                end
                    
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
                [obs_left, left_posterior, q, counter] = hardcoded_number(hardcoded_left, left_posterior, std_dev, q, obs_left, counter);
            end
        else
            if hard_coded_example == 0
               [obs_right, right_posterior, q] = sample_and_update(right_mean, right_posterior, std_dev, q, obs_right);
            else
                [obs_right, right_posterior, q, counter] = hardcoded_number(hardcoded_right, right_posterior, std_dev, q, obs_right,counter);
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

function [obs, posterior_mean, std_dev_prior, counter] = hardcoded_number(hard_coded_array, posterior_mean, std_dv, std_dev_prior, obs, counter)
    kalman_gain = std_dev_prior/(std_dv + std_dev_prior);
    sample = hard_coded_array(counter);
    posterior_mean = posterior_mean + kalman_gain*(sample - posterior_mean);
    std_dev_prior = std_dev_prior - kalman_gain*std_dev_prior;
    obs(end+1) = sample;
    counter = counter+1;
end

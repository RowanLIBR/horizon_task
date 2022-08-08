function params = load_parameters(game, trial)
   params{1} = game(trial).gameLength; % horizon
   r = game(trial).rewards;
   params{2} = r(1,:); % left rewards
   params{3} = r(2,:); % right rewards
   f = game(trial).nforced;
   s = sum(f);
   if s == 6
       params{4} = 1; % equal
   else
       params{4} = 0; % not equal
       if s == 5
           params{5} = 1; % left weighted
       else
           params{5} = 2; % right weighted
       end
   end
   params{6} = game(trial).correct;
   params{7} = game(trial).mean(1,:);
   params{8} = game(trial).mean(2,:);
   params{9} = f;
   
end
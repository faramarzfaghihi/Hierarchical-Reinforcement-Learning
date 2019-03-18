%%%%%%%%%%% MATLAB code for 'Hierarchically organized behavior and its neural foundations:
%%%%%%%%%%% A reinforcement learning perspective' Botnivick, 2009 Figure4.
%%%% Written by Faramarz Faghihi

   
     load('4RoomsSimVariables.mat')

    if ~isequal(size(I),[S,O])
        error('myApp:dimesionI', 'I should be a matrix of dimension S x O')
    end
    
    if ~isequal(length(B),O)
        error('myApp:dimesionB', 'B should be a vector of length O')
    end
    
    if ~isequal(size(T),[S,A])
        error('myApp:dimesionT', 'T should be a matrix of dimension S x A')
    end

    if ~isequal(length(R),S)
        error('myApp:dimesionR', 'B should be a vector of length S')
    end    
    
    if start < 0 || start > S 
        error('myApp:start', 'S should be a positive number and less than or equal to S')
    end
    
    if goal < 1 || goal > S
        error('myApp:goal', 'goal should be a number between 1 and S')
    end  

    % initialize variables
    max_time = 50000;  % time limit for each episode 
    init_Val = 0;   % initial value of value functions
    V = ones(S,1) * init_Val; % top-level value function
    Vo = ones(S,O) * init_Val; % value functions for each option (cols)
    W = zeros(S,A+O); % top-level action strengths. rows=states, cols=action probabilities
    Wo = zeros(O,S,A); % option-specific action strenghts
    solnTimes = zeros(1,episodes); % number of time steps the agent took to reach goal in each episode

    for episode = 1:episodes+leadTime
    
        % if no action was specified initially, randomly select one
        if start == 0,
            startState = round(rand(1)*(S-1)) + 1;
            while (startState == goal || ismember(startState,noGoStates)),
                startState = round(rand(1)*(S-1)) + 1;
            end
        else startState = start;
        end

        clear a;
        clear r;
        t = 1;
        s(t) = startState;

        % continue episode until current state is the goal state or reached time limit
        while ((s(t) ~= goal && t <= max_time) || (episode <= leadTime && t <= max_time))

            denominator = sum(exp(W(s(t),:)/tau));  % denominatorinator of equation (1) in paper for all a's
            probs = exp(W(s(t),:)/tau)/denominator;  

            %select action probabilistically according to probs
            Selected_action = 0;
            while (Selected_action == 0)
                if withOptions
                    foo = randperm(A+O); index = foo(1);
                else
                    foo = randperm(A); index = foo(1);
                end
                thresholdold = rand(1);
                if (probs(index) >= thresholdold),
                    if ((index <= A) || ((index > A) && (I(s(t),index-A)==1))),
                        a(t) = index;
                        Selected_action = 1;
                    end
                end
            end

            % if a non-primitive action (option) is selected...
            if (a(t) > A)
                op = a(t) - A;
                wm = [a(t), s(t), t];
                
                % select primitive actions until subgoal is reached
                while (s(t) ~= B(op) && t < max_time)
                    
                    denominator = sum(exp(Wo(op,s(t),:)/tau));
                    probs = exp(Wo(op,s(t),:)/tau)/denominator;
                    
                    Selected_action = 0;
                    while (Selected_action == 0)
                        foo = randperm(A); 
                        index = foo(1);
                        thresholdold = rand(1);

                        if probs(index) >= thresholdold,
                            a(t) = index;
                            Selected_action = 1;
                        end
                    end
                    
                    % update state
                    s(t+1) = T(s(t),a(t));
                    if (ismember(s(t+1),noGoStates))
                        s(t+1) = s(t);
                    end
                    
                    % get pseudoreward if current state is a termination state for the option
                    if (s(t+1) == B(op))
                        pseudor = 100; 
                    else
                        pseudor = 0;
                    end;
                    r(t+1) = pseudor;
                    
                    % compute prediction error (delta)
                    delta = r(t+1) + (ngamma * Vo(s(t+1),op)) - Vo(s(t),op);
                    
                    % update option's value function
                    Vo(s(t),op) = Vo(s(t),op) + alpha_C * delta;
                    
                    % update option's policy
                    Wo(op,s(t),a(t)) = Wo(op,s(t),a(t)) + alpha_A * delta;
                    
                    if (s(t+1) == B(op) || s(t+1) == goal || t >= max_time),
                        % get real reward (but only if also at exit state of option)
                        if (s(t+1) == B(op));
                            r(t+1) = R(s(t+1));
                        end
                        if (episode <= leadTime) 
                            r(t+1) = 0; 
                        end;
                        
                        % compute delta at top level
                        lag = t - wm(3);
                        initState = wm(2);
                        cumDis = ngamma^lag;
                        delta = (ngamma^(lag-1))*r(t+1)+ cumDis*(V(s(t+1))) - V(initState); %????why is it V instead of Vo???

                        % update root value function (for s where option chosen)
                        V(initState) = V(initState) + alpha_C * delta;
                        
                        % update root actionStrenghts (for s where o selected)
                        W(initState,wm(1)) = W(initState,wm(1)) + alpha_A * delta;
                    end

                    t = t+1;
                end


            else % primitive action selected at root level

                % update state (deteriministic)
                s(t+1) = T(s(t),a(t));
                if (ismember(s(t+1),noGoStates))
                    s(t+1) = s(t);
                end

                % get reward
                r(t+1) = R(s(t+1));
                if (episode <= leadTime) 
                    r(t+1) = 0; 
                end

                % compute delta
                delta = r(t+1) + (ngamma * V(s(t+1))) - V(s(t));
                
                % update value function
                V(s(t)) = V(s(t)) + alpha_C * delta;

                % update policy
                W(s(t),a(t)) = W(s(t),a(t)) + alpha_A * delta;

                t = t+1;

            end % (stuff following on selection of action at root level)
%             solnTimes(episode) = t;
        end % end of a single episode

        % update plots
        
        if (episode > leadTime)
            if S == 169 && A == 8 && O == 8
                subplot(1,4,1:2);hold on; plot (episode, t,'o'); drawnow;
                xlabel('episode')
                ylabel('number of time steps')
                theStrengths = reshape(Wo(option_for_plot,:,:),169,A);
                [vals, inds] = max (theStrengths,[],2);
                strengthsMatrix = reshape(inds,13,13)';
                       
            else
                hold on; plot (episode, t,'o'); drawnow;           
            end
        end
        
    end
    
    if S == 169 && A == 8 && O == 8
        U = zeros(1,169); 
        V = zeros(1,169); 
        X = zeros(1,169); 
        Y = zeros(1,169); 
        for i=1:13
            for j=1:13
                if ismember(j+13*(i-1),find(I(:,option_for_plot))) 
                    X(i+13*(j-1)) = j; 
                    Y(i+13*(j-1)) = i; 

                    switch strengthsMatrix(i,j)
                        case 1
                            u = 0; v = 1;
                        case 2
                            u = 1/sqrt(2); v = 1/sqrt(2);
                        case 3
                            u = 1; v = 0;
                        case 4
                            u = 1/sqrt(2); v = -1/sqrt(2);
                        case 5
                            u = 0; v=-1;
                        case 6
                            u = -1/sqrt(2); v = -1/sqrt(2);
                        case 7
                            u = -1; v=0;
                        case 8
                            u = -1/sqrt(2); v = 1/sqrt(2);
                    end
                    U(i+13*(j-1)) = u;
                    V((i+13*(j-1))) = v; 
                end
            end
        end
        
        walls = ones(1,169);
        for i=1:169
            if ismember(i,noGoStates)
                walls(i) = 0;
            end
        end
    
        subplot(1,4,3:4); 
        walls = reshape(walls,13,13)';
        walls(ceil(B(option_for_plot)/13),mod(B(option_for_plot),13)) = .5;
        imagesc(walls); hold on; axis square;
        xlabel(['option-specific action strengths for option ',num2str(option_for_plot),])
        quiver(X,Y,U,-V,0.25); 
        ticks = 1:13;
        set(gca,'YTick',ticks)
        set(gca,'XTick',ticks)

        borders = 1.5:12.5;
        for i=1:12
            line([0 ; 13.5],[borders(i) ; borders(i)])
            line([borders(i) ; borders(i)], [0 ; 13.5])
        end
    
        colormap(gray);
    end
    
%    figure 
%   imagesc(strengthsMatrix)
%     



    
    

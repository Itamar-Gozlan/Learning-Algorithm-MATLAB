function w = perceptron(m, d, Xtrain, Ytrain, maxupdates)
	w = zeros(1,d);
	count = 0;
	done = false;
	while ~done
		done = true;
		t = 1;
		while t <= m
			if Ytrain(t)*dot(Xtrain(t,:),w) <= 0
				% do update
				w_t = w;
				w = w_t + Ytrain(t).*Xtrain(t,:);
				% not done
				done = false;
				count = count + 1;
				if count >= maxupdates
					done = true;
					break;
				end
			end
			t = t+1;
		end
	end
end
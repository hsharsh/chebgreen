function g = green(x,y)
    g = 0.0;
    g = (x <= y).*(x .* (1-y)) + (x > y).*(y .* (1-x));
end
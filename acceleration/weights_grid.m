function weights = weights_grid(sz, numPoints)

u = rand;
coordX = ceil((sz(1) / numPoints(1)) .* ((0:numPoints(1)-1) + u));
v = rand;
coordY = ceil((sz(2) / numPoints(2)) .* ((0:numPoints(2)-1) + v));

weights = zeros(sz, 'single');
weights(coordX, coordY) = 1;

end

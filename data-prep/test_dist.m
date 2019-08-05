clc; clear;
open('phase_diagram.fig');
hold on;


x_expr = 't3_([\-|\d|\.]+)\.mat$';
y_expr = '^m_([\-|\d|\.]+)';

x = [];
y = [];

load('filenames_momentum.mat');

for i = 1: length(test_filenames)
    test_filename = convertCharsToStrings(test_filenames(i, :));
    test_filename = strtrim(test_filename);
    [~, basename, ext] = fileparts(test_filename);
    filename = strcat(basename,ext);
    [tokens, ~] = regexp(filename, x_expr, 'tokens', 'match');
    x(i) = str2double(string(tokens{1}));
    [tokens, ~] = regexp(filename, y_expr, 'tokens', 'match');
    y(i) = str2double(string(tokens{1}));
end

% p = scatter(x, y, 20, [0, 0, 1], 'filled');
p = scatter(x, y, 20, [0, 0, 1]);

legend([p], {'test samples'});
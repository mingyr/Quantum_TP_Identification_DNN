clc; clear;
open('phase_diagram.fig');
hold on;


x_expr = 't3_([\-|\d|\.]+)\.mat$';
y_expr = '^m_([\-|\d|\.]+)';

x = [];
y = [];

load('filenames_momentum.mat');

for i = 1: length(val_filenames)
    val_filename = convertCharsToStrings(val_filenames(i, :));
    val_filename = strtrim(val_filename);
    [~, basename, ext] = fileparts(val_filename);
    filename = strcat(basename,ext);
    [tokens, ~] = regexp(filename, x_expr, 'tokens', 'match');
    x(i) = str2double(string(tokens{1}));
    [tokens, ~] = regexp(filename, y_expr, 'tokens', 'match');
    y(i) = str2double(string(tokens{1}));
end

% p = scatter(x, y, 20, [0, 0, 1], 'filled');
p = scatter(x, y, 20, [0, 0, 1]);

legend([p], {'val. samples'});

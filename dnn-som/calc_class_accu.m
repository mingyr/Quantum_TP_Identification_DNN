function accus = calc_class_accu(classes, indices) 
classes = reshape(classes, [], 1);
indices = reshape(indices, [], 1);

class = unique(classes);
% disp(class);

accus = [];
for i = 1 : numel(class)
    pos = find(classes == class(i));
    % disp(['i = ', num2str(class(i)), ', count = ', num2str(numel(pos))]);

    cc = classes(pos);
    ci = indices(pos);
    accu = numel(find(ci == cc)) / numel(cc);

    % disp('accuracy =');
    % disp(accu);
    accus(i) = accu;
end



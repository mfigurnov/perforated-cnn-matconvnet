function [ res ] = vl_getfielddefault(l, field)
  if isfield(l, field)
    res = l.(field);
  else
    res = [];
  end
end
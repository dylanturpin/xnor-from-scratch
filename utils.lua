U = {}

function U.computeScore(output, target)
   -- Computes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   local top1 = correct:narrow(2, 1, 1):sum() / batchSize
   local top5 = correct:narrow(2, 1, 5):sum() / batchSize

   return top1 * 100, top5 * 100
end

return U

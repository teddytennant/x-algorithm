package com.twitter.botmaker.function.collection;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.Extractor;
import com.twitter.botmaker.compiler.exceptions.FunctionFailure;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.Constant;
import com.twitter.botmaker.runtime.Runtime;
import com.twitter.util.Function;
import com.twitter.util.Future;
import com.twitter.util.Futures;

public final class Any {
  private static final ASTNode.Signature ANY_SIGNATURE = new ASTNode.Signature(
      ImmutableList.of(Type.BOOLEAN, Type.collectionOf(Type.OBJECT)),
      ImmutableList.of(Type.LONG),
      Type.BOOLEAN
  );
  private static final Future<Boolean> FUTURE_TRUE = Future.value(Boolean.TRUE);
  private static final Future<Boolean> FUTURE_FALSE = Future.value(Boolean.FALSE);

  private Any() {
  }

  public static ASTNode createAny(String exprText,
                                  String varName,
                                  ASTNode conditionNode,
                                  long scope,
                                  ASTNode collectionNode,
                                  Optional<ASTNode> batchSizeNodeOpt) throws SemanticCheckFailure {

    ASTNode batchSizeNode = batchSizeNodeOpt.isPresent() ? batchSizeNodeOpt.get()
        : Constant.of("20", 20L);

    return new ASTNode<Runtime>(
        exprText, ImmutableList.of(conditionNode, collectionNode, batchSizeNode)) {

      @Override
      public long computeFingerprintScope(long currentScope) {
        return super.computeFingerprintScope(Math.min(currentScope, scope));
      }

      @Override
      public Signature getSignature() {
        return ANY_SIGNATURE;
      }

      @Override
      protected CacheLevel getCacheLevel() {
        return CacheLevel.Global;
      }

      @Override
      public Extractor<Runtime> toExtractor() {
        Extractor conditionExtractor = conditionNode.toExtractor();
        Extractor collectionExtractor = collectionNode.toExtractor();
        Extractor batchSizeExtractor = batchSizeNode.toExtractor();
        ImmutableList<Extractor> children =
            ImmutableList.of(conditionExtractor, collectionExtractor, batchSizeExtractor);

        if (isFunctor(children)) {
          Extractor.Functor funcFunctor = (Extractor.Functor) conditionExtractor;
          Extractor.Functor collectionFunctor = (Extractor.Functor) collectionExtractor;
          return new Extractor.Functor<Runtime>(this, children) {

            @Override
            public Future<Object> run(Context<Runtime> context) {
              try {
                return Future.value(apply(context));
              } catch (Exception ex) {
                return Future.exception(ex);
              }
            }

            @Override
            public Boolean apply(Context<Runtime> context) {
              try {
                Collection<Object> collection =
                    (Collection<Object>) collectionFunctor.apply(context);
                for (Object elem : collection) {
                  ConcurrentMap<String, Future<Object>> variableMap = new ConcurrentHashMap<>();
                  variableMap.put(varName, Future.value(elem));
                  Context newContext = Context.of(context, variableMap);
                  if ((Boolean) funcFunctor.apply(newContext)) {
                    return Boolean.TRUE;
                  }
                }
                return Boolean.FALSE;
              } catch (Exception e) {
                throw FunctionFailure.wrap(e, getAstNode(), context);
              }
            }
          };
        } else {
          return new Extractor<Runtime>(this, children) {
            @Override
            public Future run(Context<Runtime> context) {
              Future<Object> collectionExtractionFuture = context.run(collectionExtractor);
              Future<Object> batchSizeExtractionFuture = context.run(batchSizeExtractor);
              return Futures.join(collectionExtractionFuture, batchSizeExtractionFuture)
                  .flatMap(collectionAndBatchSize -> {
                        Collection<Object> collection =
                            (Collection<Object>) collectionAndBatchSize._1();
                        int batchSize = ((Long) collectionAndBatchSize._2()).intValue();
                        if (batchSize <= 0) {
                          throw new FunctionFailure(getAstNode(),
                              context.getStackFrames(),
                              new IllegalArgumentException(
                                  "batch size must be a positive integer"));
                        }

                        if (collection.isEmpty()) {
                          return FUTURE_FALSE;
                        }

                        return evaluate(
                            context,
                            varName,
                            new ArrayList<>(collection),
                            0,
                            conditionExtractor,
                            batchSize
                        );
                      }
                  ).rescue(Function.func((Throwable ex) ->
                      Future.exception(FunctionFailure.wrap(ex, getAstNode(), context))));
            }
          };
        }
      }
    };

  }

  private static Future<Boolean> evaluate(Context<Runtime> context, String varName,
                                          List<Object> originatingCollection,
                                          int startIdx,
                                          Extractor conditionExtractor,
                                          int batchSize) {

    int remainingElements = originatingCollection.size() - startIdx;
    int elementsThisRound = Math.min(remainingElements, batchSize);

    List<Future<Boolean>> listOfFutures = new ArrayList<>(elementsThisRound);
    for (int i = 0; i < elementsThisRound; ++i) {
      ConcurrentMap<String, Future<Object>> variableMap = new ConcurrentHashMap<>();
      variableMap.put(varName, Future.value(originatingCollection.get(startIdx + i)));
      Context<?> newContext = Context.of(context, variableMap);
      listOfFutures.add((Future<Boolean>) newContext.run(conditionExtractor));
    }
    Future<List<Boolean>> batchResponses = Futures.collect(listOfFutures);
    return batchResponses.flatMap(nextVals -> {
      if (nextVals.contains(Boolean.TRUE)) {
        return FUTURE_TRUE;
      }
      if (remainingElements == elementsThisRound) {
        return FUTURE_FALSE;
      }
      return evaluate(context, varName, originatingCollection,
          startIdx + elementsThisRound, conditionExtractor, batchSize);
    });
  }

}

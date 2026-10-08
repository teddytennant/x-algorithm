package com.twitter.botmaker.function.collection;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentMap;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.Extractor;
import com.twitter.botmaker.Extractor.Functor;
import com.twitter.botmaker.compiler.exceptions.FunctionFailure;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.runtime.Runtime;
import com.twitter.util.Function;
import com.twitter.util.Future;

public abstract class ForEach extends ASTNode<Runtime> {

  private static final Type TP1 = Type.newGenericType();
  private static final Type TP2 = Type.newGenericType();

  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.collectionOf(TP1), TP2),
      Type.listOf(TP2)
  );

  private static final int MAX_CONCURRENCY = 50;

  private final String variableName;

  public ForEach(
      String exprText,
      String variableName,
      ASTNode collectionNode,
      ASTNode bodyNode) throws SemanticCheckFailure {
    super(exprText, ImmutableList.of(collectionNode, bodyNode));
    this.variableName = variableName;
  }

  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  @Override
  public Extractor<Runtime> toExtractor() {
    List<Extractor> childrenExtractors = buildExtractorsOfChildren();
    Extractor collectionExtractor = childrenExtractors.get(0);
    Extractor bodyExtractor = childrenExtractors.get(1);
    ImmutableList<Extractor> children = ImmutableList.of(collectionExtractor, bodyExtractor);

    if (isFunctor(children)) {
      Functor collectionFunctor = (Functor) collectionExtractor;
      Functor bodyFunctor = (Functor) bodyExtractor;
      return new Functor<Runtime>(this, children) {

        @Override
        public Future<Object> run(Context<Runtime> context) {
          try {
            return Future.value(apply(context));
          } catch (Exception ex) {
            return Future.exception(ex);
          }
        }

        @Override
        public Object apply(Context<Runtime> context) {
          try {
            Collection<Object> collection = (Collection<Object>) collectionFunctor.apply(context);
            List<Object> resultList = new ArrayList<>(collection.size());
            for (Object elem : collection) {
              ConcurrentMap<String, Future<Object>> variableMap = Maps.newConcurrentMap();
              variableMap.put(variableName, Future.value(elem));
              Context newContext = Context.of(context, variableMap);
              resultList.add(bodyFunctor.apply(newContext));
            }
            return Collections.unmodifiableList(resultList);
          } catch (Exception ex) {
            throw FunctionFailure.wrap(ex, getAstNode(), context);
          }
        }
      };

    } else {
      return new Extractor<Runtime>(this, children) {
        @Override
        public Future run(Context<Runtime> context) {
          return ForEach.this.run(context, collectionExtractor, bodyExtractor)
              .rescue(Function.func((Exception ex) ->
                  Future.exception(FunctionFailure.wrap(ex, getAstNode(), context))));
        }
      };
    }
  }

  private Future run(
      Context<Runtime> context, Extractor collectionExtractor, Extractor bodyExtractor) {
    return context.run(collectionExtractor).flatMap((Object collectionObj) -> {
          Collection<Object> collection = (Collection<Object>) collectionObj;
          if (collection.size() <= MAX_CONCURRENCY) {
            return extractUnderConcurrentCase(context, collection, bodyExtractor);
          }
          return extractLoopingCase(context, new ArrayList<>(collection), 0,
              bodyExtractor, ImmutableList.of());
        }
    );
  }

  private Future extractUnderConcurrentCase(Context<Runtime> context,
                                            Collection<Object> collection,
                                            Extractor bodyExtractor) {
    List<Future<Object>> listResultFutures = new ArrayList<>(collection.size());
    for (Object elem : collection) {
      ConcurrentMap<String, Future<Object>> variableMap = Maps.newConcurrentMap();
      variableMap.put(variableName, Future.value(elem));
      Context newContext = Context.of(context, variableMap);
      listResultFutures.add(newContext.run(bodyExtractor));
    }
    return Future.collect(listResultFutures);
  }

  private Future extractLoopingCase(Context<Runtime> context,
                                    List<Object> originatingCollection,
                                    int startIdx,
                                    Extractor bodyExtractor,
                                    List<Object> previousCollectedObjects) {

    int remainingElements = originatingCollection.size() - startIdx;
    int elementsThisRound = Math.min(remainingElements, MAX_CONCURRENCY);

    List<Future<Object>> batchFuturesList = new ArrayList<>(elementsThisRound);
    for (int i = 0; i < elementsThisRound; ++i) {
      ConcurrentMap<String, Future<Object>> variableMap = Maps.newConcurrentMap();
      variableMap.put(variableName, Future.value(originatingCollection.get(startIdx + i)));
      Context newContext = Context.of(context, variableMap);
      batchFuturesList.add(newContext.run(bodyExtractor));
    }

    Future<List<Object>> currentCollectedObjectsFuture = Future.collect(batchFuturesList)
        .map(thisBatchObjects -> combine(previousCollectedObjects, thisBatchObjects));

    if (remainingElements == elementsThisRound) {
      return currentCollectedObjectsFuture;
    }

    return currentCollectedObjectsFuture.flatMap(currentCollectedObjects ->
        extractLoopingCase(context, originatingCollection, startIdx + elementsThisRound,
            bodyExtractor, currentCollectedObjects));
  }

  private static <T> List<T> combine(List<? extends T> first, List<? extends T> second) {
    List<T> combined = new ArrayList<>(first.size() + second.size());
    combined.addAll(first);
    combined.addAll(second);
    return combined;
  }

  public static ForEach of(
      String exprText,
      String variableName,
      long scope,
      ASTNode collectionNode,
      ASTNode bodyNode) throws SemanticCheckFailure {

    return new ForEach(exprText, variableName, collectionNode, bodyNode) {
      @Override
      public long computeFingerprintScope(long currentScope) {
        return super.computeFingerprintScope(Math.min(currentScope, scope));
      }
    };
  }
}

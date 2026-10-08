package com.twitter.botmaker.function.collection;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentMap;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
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

public abstract class Filter extends ASTNode<Runtime> {

  private static final Type TP = Type.newGenericType();
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.BOOLEAN, Type.collectionOf(TP)),
      Type.listOf(TP)
  );

  private static final int MAX_CONCURRENCY = 50;

  private final String variableName;
  private final ASTNode conditionNode;
  private final ASTNode collectionNode;

  public Filter(
      String exprText,
      String variableName,
      ASTNode conditionNode,
      ASTNode collectionNode
  ) throws SemanticCheckFailure {
    super(exprText, ImmutableList.of(conditionNode, collectionNode));
    this.variableName = variableName;
    this.conditionNode = conditionNode;
    this.collectionNode = collectionNode;
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
    Extractor conditionExtractor = conditionNode.toExtractor();
    Extractor collectionExtractor = collectionNode.toExtractor();
    ImmutableList<Extractor> children = ImmutableList.of(conditionExtractor, collectionExtractor);

    if (isFunctor(children)) {
      Functor conditionFunctor = (Functor) conditionExtractor;
      Functor collectionFunctor = (Functor) collectionExtractor;

      return new Functor<Runtime>(this, children) {

        @Override
        public Future run(Context<Runtime> context) {
          return Filter.this.run(context, conditionExtractor, collectionExtractor);
        }

        @Override
        public Object apply(Context<Runtime> context) {
          try {
            Collection<Object> collection = (Collection<Object>) collectionFunctor.apply(context);
            ArrayList<Object> filtered = new ArrayList<>();
            for (Object elem : collection) {
              ConcurrentMap<String, Future<Object>> variableMap = Maps.newConcurrentMap();
              variableMap.put(variableName, Future.value(elem));
              Context newContext = Context.of(context, variableMap);

              Boolean cond = (Boolean) conditionFunctor.apply(newContext);
              if (cond) {
                filtered.add(elem);
              }
            }
            filtered.trimToSize();
            return Collections.unmodifiableList(filtered);
          } catch (Exception e) {
            throw FunctionFailure.wrap(e, getAstNode(), context);
          }
        }
      };
    } else {
      return new Extractor<Runtime>(this, children) {
        @Override
        public Future run(Context<Runtime> context) {
          return Filter.this.run(context, conditionExtractor, collectionExtractor);
        }
      };
    }
  }

  private Future run(
      Context<Runtime> context, Extractor conditionExtractor, Extractor collectionExtractor) {
    return context.run(collectionExtractor).flatMap((Object collectionObj) -> {
      @SuppressWarnings("unchecked") final Collection<Object> collection =
          (Collection<Object>) collectionObj;

      if (collection.size() <= MAX_CONCURRENCY) {
        return extractWithoutBatching(context, conditionExtractor, collection);
      }
      return extractWithBatching(context, conditionExtractor, new ArrayList<>(collection),
          0, ImmutableList.of());
    }).rescue(Function.func((Throwable ex) ->
        Future.exception(FunctionFailure.wrap(ex, this, context))));
  }

  private Future extractWithoutBatching(Context<Runtime> context,
                                        Extractor conditionExtractor,
                                        Collection<Object> collection) {

    List<Future<Object>> conditions =
        Lists.newArrayListWithCapacity(collection.size());

    List<Object> values =
        Lists.newArrayListWithCapacity(collection.size());
    for (Object elem : collection) {
      ConcurrentMap<String, Future<Object>> variableMap = Maps.newConcurrentMap();
      variableMap.put(variableName, Future.value(elem));
      Context newContext = Context.of(context, variableMap);
      Future<Object> cond = newContext.run(conditionExtractor);
      conditions.add(cond);
      values.add(elem);
    }
    return Future.collect(conditions).map((List<Object> condResults) -> {
      ArrayList<Object> filtered = new ArrayList<>();
      for (int i = 0; i < condResults.size(); i++) {
        if ((Boolean) condResults.get(i)) {
          filtered.add(values.get(i));
        }
      }
      filtered.trimToSize();
      return Collections.unmodifiableList(filtered);
    });
  }

  private Future extractWithBatching(Context<Runtime> context,
                                     Extractor conditionExtractor,
                                     List<Object> originatingCollection,
                                     int startIdx,
                                     List<Object> previousFilteredObjects) {

    int remainingElements = originatingCollection.size() - startIdx;
    int elementsThisRound = Math.min(remainingElements, MAX_CONCURRENCY);

    List<Future<Object>> batchConditionsFuturesList =
        Lists.newArrayListWithCapacity(elementsThisRound);
    List<Object> batchValuesList =
        Lists.newArrayListWithCapacity(elementsThisRound);
    for (int i = 0; i < elementsThisRound; ++i) {
      ConcurrentMap<String, Future<Object>> variableMap = Maps.newConcurrentMap();
      variableMap.put(variableName, Future.value(originatingCollection.get(startIdx + i)));
      Context newContext = Context.of(context, variableMap);
      Future<Object> cond = newContext.run(conditionExtractor);
      batchConditionsFuturesList.add(cond);
      batchValuesList.add(originatingCollection.get(startIdx + i));
    }

    Future<List<Object>> currentFilteredObjectsFuture = Future.collect(
        batchConditionsFuturesList)
        .map((List<Object> thisBatchCondResults) -> {
          List<Object> thisBatchFilteredObjects = new ArrayList<>();
          for (int i = 0; i < thisBatchCondResults.size(); i++) {
            if ((Boolean) thisBatchCondResults.get(i)) {
              thisBatchFilteredObjects.add(batchValuesList.get(i));
            }
          }
          return combine(previousFilteredObjects, thisBatchFilteredObjects);
        });

    if (remainingElements == elementsThisRound) {
      return currentFilteredObjectsFuture;
    }

    return currentFilteredObjectsFuture.flatMap(currentFilteredObjects ->
        extractWithBatching(context, conditionExtractor, originatingCollection,
            startIdx + elementsThisRound, currentFilteredObjects));
  }

  private static <T> List<T> combine(List<? extends T> first, List<? extends T> second) {
    List<T> combined = new ArrayList<>(first.size() + second.size());
    combined.addAll(first);
    combined.addAll(second);
    return combined;
  }

  public static Filter of(
      String exprText,
      String variableName,
      ASTNode conditionNode,
      long scope,
      ASTNode collectionNode) throws SemanticCheckFailure {

    return new Filter(exprText, variableName, conditionNode, collectionNode) {
      @Override
      public long computeFingerprintScope(long currentScope) {
        return super.computeFingerprintScope(Math.min(currentScope, scope));
      }
    };
  }
}

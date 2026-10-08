package com.twitter.botmaker.function.logic;

import java.util.ArrayList;
import java.util.List;

import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.Extractor;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.ContextCancelledException;
import com.twitter.botmaker.compiler.exceptions.FunctionFailure;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.util.Function;
import com.twitter.util.Future;
import com.twitter.util.Futures;
import com.twitter.util.Try;

@BotMakerFunction(
    name = {"BatchedOr"},
    argTypes = {"Boolean", "[Boolean...]"},
    arguments = {"Boolean", "0 or more additional Booleans"},
    returnType = "Boolean",
    description = "Provides an OR operator over 1+ elements. No guarantee on short circuiting - "
        + "BatchedOr can evaluate many elements underneath before returning, even if the first "
        + "argument passed in evaluates to TRUE.",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "BatchedOr(FALSE, TRUE, FALSE)"
    }
)
public class BatchedOr extends ASTNode<Runtime> {

  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.BOOLEAN),
      Type.BOOLEAN,
      Type.BOOLEAN
  );

  public BatchedOr(String exprText, ImmutableList<ASTNode> children) throws SemanticCheckFailure {
    super(exprText, children);
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  public Extractor<Runtime> toExtractor() {
    ImmutableList<Extractor> childObjectExtractors = buildExtractorsOfChildren();
    if (isFunctor(childObjectExtractors)) {
      return new Extractor.FunctionExtractor<Runtime>(this, childObjectExtractors) {
        @Override
        public Future<Object> evaluate(Context<Runtime> context, List<Object> args) {
          try {
            for (Object arg : args) {
              if (context.isCancelled()) {
                throw new ContextCancelledException();
              }
              if ((Boolean) arg) {
                return Future.True();
              }
            }
            return Future.False();
          } catch (Exception e) {
            return Future.exception(FunctionFailure.wrap(e, getAstNode(), context));
          }
        }

        @Override
        public Boolean apply(Context<Runtime> context) throws Exception {
          try {
            for (Extractor extractor : childObjectExtractors) {
              if (context.isCancelled()) {
                throw new ContextCancelledException();
              }
              Extractor.Functor extractorFunctor = (Extractor.Functor) extractor;
              if ((Boolean) extractorFunctor.apply(context)) {
                return Boolean.TRUE;
              }
            }
            return Boolean.FALSE;
          } catch (Exception e) {
            throw FunctionFailure.wrap(e, getAstNode(), context);
          }
        }
      };
    }
    return new BatchedOrExtractor(this, childObjectExtractors);
  }

  private static class BatchedOrExtractor extends Extractor<Runtime> {

    private static final int MICRO_BATCH_SIZE = 20;

    public BatchedOrExtractor(ASTNode astNode, ImmutableList<Extractor> children) {
      super(astNode, children);
    }

    @Override
    public Future<Object> run(Context<Runtime> context) {
      return evaluate(context, getChildren(), 0)
          .rescue(Function.func((Throwable ex) ->
              Future.exception(FunctionFailure.wrap(ex, getAstNode(), context))));
    }

    private static Future<Object> evaluate(Context<Runtime> context,
                                           List<Extractor> childExtractors,
                                           int startIdx) {

      if (context.isCancelled()) {
        return Future.exception(new ContextCancelledException());
      }
      int remainingElements = childExtractors.size() - startIdx;
      int elementsThisRound = Math.min(remainingElements, MICRO_BATCH_SIZE);
      List<Future<Try<Boolean>>> listOfFutures = new ArrayList<>(elementsThisRound);
      for (int i = 0; i < elementsThisRound; ++i) {
        Extractor extractor = childExtractors.get(startIdx + i);
        Future<Try<Boolean>> liftedFuture = ((Future<Boolean>) extractor.run(context)).liftToTry();
        listOfFutures.add(liftedFuture);
      }
      Future<List<Try<Boolean>>> batchResponses = Futures.collect(listOfFutures);
      return batchResponses.flatMap(nextVals -> {
        for (Try<Boolean> result : nextVals) {
          if (result.get()) {
            return Future.True();
          }
        }
        if (remainingElements == elementsThisRound) {
          return Future.False();
        }
        return evaluate(context, childExtractors, startIdx + elementsThisRound);
      });
    }


  }


}

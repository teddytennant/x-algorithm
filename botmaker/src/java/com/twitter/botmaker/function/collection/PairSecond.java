package com.twitter.botmaker.function.collection;

import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.tuples.Pair;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode1;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    argTypes = {"Pair<A, B>"},
    arguments = {"pair"},
    returnType = "B",
    description = "Returns the second of a given pair",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "PairSecond(ToPair(\"asdf\", 123))"
    }
)
public class PairSecond extends FunctionNode1<Runtime, Pair> {

  private static final Type TPA = Type.newGenericType();
  private static final Type TPB = Type.newGenericType();
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.pairOf(TPA, TPB)),
      TPB
  );

  public PairSecond(String exprText, ImmutableList<ASTNode> children)
      throws SemanticCheckFailure {
    super(exprText, children);
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
  @SuppressWarnings("unchecked")
  protected Object apply(Context<Runtime> context, Pair pair) {
    return pair.getSecond();
  }
}

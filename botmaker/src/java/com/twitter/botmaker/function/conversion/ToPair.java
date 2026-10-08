package com.twitter.botmaker.function.conversion;

import java.util.List;

import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.tuples.Pair;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode2;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    name = {
        "Pair",
        "ToPair"
    },
    argTypes = {
        "A",
        "B"
    },
    arguments = {
        "first",
        "second"
    },
    returnType = "Pair<A, B>",
    description = "Makes a pair of values",
    actionLevel = ActionLevel.NO_ACTION
)
public class ToPair extends FunctionNode2<Runtime, Object, Object> {

  private static final Type TPA = Type.newGenericType();
  private static final Type TPB = Type.newGenericType();
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(TPA, TPB),
      Type.pairOf(TPA, TPB));

  public ToPair(String exprText, ImmutableList<ASTNode> children)
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

  public static ToPair of(
      String exprText, List<ASTNode> children) throws SemanticCheckFailure {
    return new ToPair(exprText, ImmutableList.copyOf(children));
  }

  public static ToPair of(
      String exprText, ASTNode... children) throws SemanticCheckFailure {
    return new ToPair(exprText, ImmutableList.copyOf(children));
  }

  @Override
  @SuppressWarnings("unchecked")
  protected Object apply(Context<Runtime> context, Object arg1, Object arg2) {
    return new Pair(arg1, arg2);
  }

  public static ImmutableList<ASTNode> convertNodesToPairs(
      ImmutableList<ASTNode> nodes, int startOffset) throws SemanticCheckFailure {
    ImmutableList.Builder<ASTNode> builder = ImmutableList.builder();
    int n = nodes.size();
    if (n < startOffset
        || (n - startOffset) % 2 != 0) {
      throw new SemanticCheckFailure(String.format(
          "Cannot convert nodes to pairs: "
              + "offset = %d, nodes to convert = %d (%s)",
          startOffset - 1,
          n - startOffset,
          n < startOffset ? "negative" : "not even"));
    }
    for (int i = 0; i < startOffset; ++i) {
      builder.add(nodes.get(i));
    }
    for (int i = startOffset; i < n; i += 2) {
      builder.add(new ToPair(
          "ToPair", ImmutableList.of(nodes.get(i), nodes.get(i + 1))));
    }
    return builder.build();
  }
}

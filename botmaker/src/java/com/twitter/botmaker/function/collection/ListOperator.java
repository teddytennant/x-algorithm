package com.twitter.botmaker.function.collection;

import java.util.List;

import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    name = {"List"},
    argTypes = {"[T...]"},
    arguments = {"0 or more elements"},
    returnType = "List<T>",
    description = "Returns a new list containing the given elements.",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "[11,22]",
        "[1,2,3,4,5,998, TRUE]",
        "[ 256, 3.5,  \"asfasdf\"  ]",
        "[1,2,3,4,5,998]"
    }
)
public class ListOperator extends FunctionNode<Runtime> {
  private static final Type TP = Type.newGenericType();
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(),
      TP,
      Type.listOf(TP)
  );

  public ListOperator(String exprText, ImmutableList<ASTNode> children)
      throws SemanticCheckFailure {
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
  protected Object apply(Context<Runtime> context, List<Object> args) {
    return args;
  }
}

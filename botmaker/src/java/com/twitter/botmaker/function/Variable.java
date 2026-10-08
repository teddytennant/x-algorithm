package com.twitter.botmaker.function;

import scala.PartialFunction;
import scala.Unit;

import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.Extractor;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.runtime.Runtime;
import com.twitter.util.Future;

public class Variable extends ASTNode<Runtime> {

  private static final Type TP = Type.newGenericType();
  private static final Signature SIGNATURE = new Signature(ImmutableList.of(TP), TP);

  public Variable(String variableText, ASTNode variableASTNode) {
    super(variableText,
        ImmutableList.of(variableASTNode),
        SIGNATURE,
        variableASTNode.getReturnType(),
        variableASTNode.getFingerprint().getBytes());
  }

  public final String getVariableText() {
    return getExprText();
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
    return new Extractor<Runtime>(this, ImmutableList.of()) {

      @Override
      public Future run(Context<Runtime> context) {
        return context.getVariable(getVariableText());
      }
    };
  }

  @Override
  public void visit(PartialFunction<ASTNode, Unit> visitor) {
  }
}
